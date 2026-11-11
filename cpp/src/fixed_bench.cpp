// Check and benchmark the integer-only CNN.
//
// Runs the hand-written network on the benchmark samples, compares its int8
// logits with the ones ONNX Runtime produced for the same quantised model,
// and reports per-prediction latency. Prints one JSON object on stdout.
//
// Usage:
//   lob_fixed --model CNN.fixed.bin --samples SAMPLES.bin [--iters 5000] [--warmup 500]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "lob/fixed_net.hpp"
#include "lob/latency_stats.hpp"
#include "lob/samples.hpp"

namespace {

// Command-line options with their defaults.
struct Options {
    std::string model_path;
    std::string samples_path;
    std::size_t iters = 5000;
    std::size_t warmup = 500;
};

// Parses the command line; throws std::invalid_argument on bad input.
Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        // Returns the value following a flag, or fails if it is missing.
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + arg);
            }
            return argv[++i];
        };

        if (arg == "--model") {
            opts.model_path = value();
        } else if (arg == "--samples") {
            opts.samples_path = value();
        } else if (arg == "--iters") {
            opts.iters = std::stoul(value());
        } else if (arg == "--warmup") {
            opts.warmup = std::stoul(value());
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (opts.model_path.empty() || opts.samples_path.empty() || opts.iters == 0) {
        throw std::invalid_argument("--model and --samples are required, --iters must be positive");
    }
    return opts;
}

// How closely the integer network reproduces ONNX Runtime on the samples.
struct Agreement {
    double accuracy = 0.0;            // fraction of samples matching the true label
    double same_class = 0.0;          // fraction where both pick the same class
    double identical_logits = 0.0;    // fraction where all three logits are bit-identical
    int max_logit_difference = 0;     // largest |difference| of any logit, in int8 steps
};

// Index of the largest of the kFixedNumClasses logits.
std::int32_t argmax(const std::int8_t* logits) {
    return static_cast<std::int32_t>(
        std::max_element(logits, logits + lob::kFixedNumClasses) - logits);
}

// Scores every sample against the true labels and the reference logits.
Agreement measure_agreement(lob::FixedNet& net, const lob::Samples& samples) {
    std::size_t correct = 0;
    std::size_t same_class = 0;
    std::size_t identical = 0;
    Agreement result;

    for (std::size_t i = 0; i < samples.count; ++i) {
        const std::int8_t* logits = net.predict(samples.window_data(i));
        const std::int8_t* reference = net.reference_logits.data() + i * lob::kFixedNumClasses;

        correct += argmax(logits) == samples.labels[i] ? 1 : 0;
        same_class += argmax(logits) == argmax(reference) ? 1 : 0;
        identical += std::equal(logits, logits + lob::kFixedNumClasses, reference) ? 1 : 0;
        for (std::size_t c = 0; c < lob::kFixedNumClasses; ++c) {
            result.max_logit_difference =
                std::max(result.max_logit_difference, std::abs(logits[c] - reference[c]));
        }
    }

    const double n = static_cast<double>(samples.count);
    result.accuracy = static_cast<double>(correct) / n;
    result.same_class = static_cast<double>(same_class) / n;
    result.identical_logits = static_cast<double>(identical) / n;
    return result;
}

// Times `iters` predictions. The timed region covers the whole path from raw
// floats to logits, including input normalisation and quantisation.
std::vector<std::int64_t> measure_latency(lob::FixedNet& net, const lob::Samples& samples,
                                          const Options& opts) {
    using Clock = std::chrono::steady_clock;
    std::vector<std::int64_t> latencies_ns;
    latencies_ns.reserve(opts.iters);

    // Accumulating the outputs keeps the compiler from optimising the work away.
    std::int64_t checksum = 0;
    for (std::size_t i = 0; i < opts.warmup + opts.iters; ++i) {
        const float* window = samples.window_data(i % samples.count);

        const auto start = Clock::now();
        const std::int8_t* logits = net.predict(window);
        const auto stop = Clock::now();

        checksum += logits[0];
        if (i >= opts.warmup) {
            latencies_ns.push_back(
                std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
        }
    }
    if (checksum == INT64_MIN) {  // never true; makes `checksum` observable
        std::abort();
    }
    return latencies_ns;
}

// The whole program: load, verify against ONNX Runtime, time, print JSON.
int run(int argc, char** argv) {
    const Options opts = parse_args(argc, argv);
    lob::FixedNet net = lob::load_fixed_net(opts.model_path);
    const lob::Samples samples = lob::load_samples(opts.samples_path);

    // The model file must describe the same input shape and sample set.
    if (samples.window != net.window || samples.n_features != net.n_features ||
        net.reference_logits.size() != samples.count * lob::kFixedNumClasses) {
        throw std::runtime_error("model file and samples file do not belong together");
    }

    const Agreement agreement = measure_agreement(net, samples);
    const lob::LatencyStats stats = lob::summarize(measure_latency(net, samples, opts));

    // Machine-readable result, stored as results/fixed_point.json.
    std::printf(
        "{\"samples\": %zu, \"accuracy\": %.6f, \"same_class_as_onnxruntime\": %.6f, "
        "\"identical_logits\": %.6f, \"max_logit_difference_lsb\": %d, \"iters\": %zu, "
        "\"mean_ns\": %.1f, \"min_ns\": %lld, \"p50_ns\": %lld, \"p90_ns\": %lld, "
        "\"p99_ns\": %lld, \"p999_ns\": %lld, \"max_ns\": %lld}\n",
        samples.count, agreement.accuracy, agreement.same_class, agreement.identical_logits,
        agreement.max_logit_difference, stats.count, stats.mean_ns,
        static_cast<long long>(stats.min_ns), static_cast<long long>(stats.p50_ns),
        static_cast<long long>(stats.p90_ns), static_cast<long long>(stats.p99_ns),
        static_cast<long long>(stats.p999_ns), static_cast<long long>(stats.max_ns));
    return 0;
}

}  // namespace

// Entry point: report any error on stderr and exit non-zero.
int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
