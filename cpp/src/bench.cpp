// Latency benchmark for an ONNX order book model.
//
// Replays real test windows through ONNX Runtime and reports the per-call
// latency distribution and throughput as one JSON object on stdout.
//
// Usage:
//   lob_bench --model MODEL.onnx --samples SAMPLES.bin
//             [--batch 1] [--iters 20000] [--warmup 2000] [--threads 1] [--no-opt]

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "lob/latency_stats.hpp"
#include "lob/samples.hpp"

namespace {

// Tensor names and output size; must match lob/export.py.
constexpr const char* kInputName = "book_window";
constexpr const char* kOutputName = "logits";
constexpr std::size_t kNumClasses = 3;

// Command-line options with their defaults.
struct Options {
    std::string model_path;
    std::string samples_path;
    std::size_t batch = 1;
    std::size_t iters = 20000;
    std::size_t warmup = 2000;
    int threads = 1;
    bool optimize = true;  // ONNX Runtime graph optimisations (fusion, constant folding, ...)
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
        } else if (arg == "--batch") {
            opts.batch = std::stoul(value());
        } else if (arg == "--iters") {
            opts.iters = std::stoul(value());
        } else if (arg == "--warmup") {
            opts.warmup = std::stoul(value());
        } else if (arg == "--threads") {
            opts.threads = std::stoi(value());
        } else if (arg == "--no-opt") {
            opts.optimize = false;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    // Validate the result.
    if (opts.model_path.empty() || opts.samples_path.empty()) {
        throw std::invalid_argument("--model and --samples are required");
    }
    if (opts.batch == 0 || opts.iters == 0) {
        throw std::invalid_argument("--batch and --iters must be positive");
    }
    return opts;
}

// Owns an ONNX Runtime session together with pre-allocated input and output
// tensors, so that this wrapper allocates nothing per call. (What ONNX Runtime
// allocates internally during Run() is outside our control.)
class Model {
public:
    Model(const Options& opts, const lob::Samples& samples)
        : batch_(opts.batch),
          window_size_(samples.window_size()),
          input_(opts.batch * samples.window_size()),
          output_(opts.batch * kNumClasses) {
        // Single-threaded session with the requested optimisation level.
        Ort::SessionOptions session_opts;
        session_opts.SetIntraOpNumThreads(opts.threads);
        session_opts.SetInterOpNumThreads(1);
        session_opts.SetGraphOptimizationLevel(opts.optimize ? GraphOptimizationLevel::ORT_ENABLE_ALL
                                                             : GraphOptimizationLevel::ORT_DISABLE_ALL);
        session_ = Ort::Session(env_, opts.model_path.c_str(), session_opts);

        // The tensors below are views over input_ / output_; ONNX Runtime
        // reads from and writes into our buffers directly.
        const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<std::int64_t, 3> input_shape = {static_cast<std::int64_t>(batch_),
                                                         static_cast<std::int64_t>(samples.window),
                                                         static_cast<std::int64_t>(samples.n_features)};
        const std::array<std::int64_t, 2> output_shape = {static_cast<std::int64_t>(batch_),
                                                          static_cast<std::int64_t>(kNumClasses)};
        input_tensor_ = Ort::Value::CreateTensor<float>(memory, input_.data(), input_.size(),
                                                        input_shape.data(), input_shape.size());
        output_tensor_ = Ort::Value::CreateTensor<float>(memory, output_.data(), output_.size(),
                                                         output_shape.data(), output_shape.size());
    }

    // Copies `batch` consecutive windows, starting at `first`, into the input tensor.
    void load_input(const lob::Samples& samples, std::size_t first) {
        std::memcpy(input_.data(), samples.window_data(first),
                    batch_ * window_size_ * sizeof(float));
    }

    // Runs inference on the loaded input; logits land in the output buffer.
    void run() {
        session_.Run(Ort::RunOptions{nullptr}, &kInputName, &input_tensor_, 1, &kOutputName,
                     &output_tensor_, 1);
    }

    // Predicted class of row `row` of the last run.
    std::int32_t predicted_class(std::size_t row) const {
        const float* logits = output_.data() + row * kNumClasses;
        return static_cast<std::int32_t>(std::max_element(logits, logits + kNumClasses) - logits);
    }

private:
    // Buffers are declared before the tensors that point into them.
    std::size_t batch_;
    std::size_t window_size_;
    std::vector<float> input_;
    std::vector<float> output_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "lob_bench"};
    Ort::Session session_{nullptr};
    Ort::Value input_tensor_{nullptr};
    Ort::Value output_tensor_{nullptr};
};

// Fraction of samples whose predicted class matches the true label.
// Python computes the same number at export time, so this doubles as an
// end-to-end check that the C++ pipeline feeds the model correctly.
double measure_accuracy(Model& model, const lob::Samples& samples, std::size_t batch) {
    std::size_t correct = 0;
    std::size_t total = 0;
    for (std::size_t first = 0; first + batch <= samples.count; first += batch) {
        model.load_input(samples, first);
        model.run();
        for (std::size_t row = 0; row < batch; ++row) {
            correct += model.predicted_class(row) == samples.labels[first + row] ? 1 : 0;
        }
        total += batch;
    }
    return static_cast<double>(correct) / static_cast<double>(total);
}

// Times `iters` inference calls; only Model::run() is inside the timed region.
std::vector<std::int64_t> measure_latency(Model& model, const lob::Samples& samples,
                                          const Options& opts) {
    using Clock = std::chrono::steady_clock;
    const std::size_t n_starts = samples.count - opts.batch + 1;

    std::vector<std::int64_t> latencies_ns;
    latencies_ns.reserve(opts.iters);

    for (std::size_t i = 0; i < opts.warmup + opts.iters; ++i) {
        // Cycle through the samples so the model sees varied, realistic inputs.
        model.load_input(samples, (i * opts.batch) % n_starts);

        const auto start = Clock::now();
        model.run();
        const auto stop = Clock::now();

        if (i >= opts.warmup) {
            latencies_ns.push_back(
                std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
        }
    }
    return latencies_ns;
}

// The whole benchmark: load inputs, check accuracy, time inference, print JSON.
int run(int argc, char** argv) {
    const Options opts = parse_args(argc, argv);
    const lob::Samples samples = lob::load_samples(opts.samples_path);
    if (samples.count < opts.batch) {
        throw std::invalid_argument("--batch is larger than the number of samples");
    }

    // Accuracy pass first (it also warms the model up), then the timed loop.
    Model model(opts, samples);
    const double accuracy = measure_accuracy(model, samples, opts.batch);
    const lob::LatencyStats stats = lob::summarize(measure_latency(model, samples, opts));

    // Throughput in windows per second, from the time spent inside inference.
    const double total_seconds = stats.mean_ns * static_cast<double>(stats.count) * 1e-9;
    const double throughput = static_cast<double>(opts.batch * stats.count) / total_seconds;

    // Machine-readable result, parsed by lob/bench.py.
    std::printf(
        "{\"batch\": %zu, \"threads\": %d, \"graph_optimizations\": %s, \"iters\": %zu, "
        "\"accuracy\": %.6f, \"mean_ns\": %.1f, \"min_ns\": %lld, \"p50_ns\": %lld, "
        "\"p90_ns\": %lld, \"p99_ns\": %lld, \"p999_ns\": %lld, \"max_ns\": %lld, "
        "\"throughput_per_s\": %.1f}\n",
        opts.batch, opts.threads, opts.optimize ? "true" : "false", stats.count, accuracy,
        stats.mean_ns, static_cast<long long>(stats.min_ns), static_cast<long long>(stats.p50_ns),
        static_cast<long long>(stats.p90_ns), static_cast<long long>(stats.p99_ns),
        static_cast<long long>(stats.p999_ns), static_cast<long long>(stats.max_ns), throughput);
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
