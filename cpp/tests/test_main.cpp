// Unit tests for the header-only helpers. Deliberately framework-free:
// a failed CHECK prints its location and makes the process exit non-zero.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lob/fixed_net.hpp"
#include "lob/latency_stats.hpp"
#include "lob/samples.hpp"

namespace {

// Number of failed CHECKs across all tests.
int failures = 0;

// Records a failure (with file and line) if `cond` is false, then keeps going.
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                      \
        }                                                                    \
    } while (false)

// True if calling `fn` throws an exception of type E.
template <typename E, typename Fn>
bool throws(Fn&& fn) {
    try {
        fn();
    } catch (const E&) {
        return true;
    }
    return false;
}

// Percentiles of the values 1..100 are easy to verify by hand.
void test_percentile_nearest_rank() {
    std::vector<std::int64_t> values(100);
    for (int i = 0; i < 100; ++i) {
        values[i] = i + 1;  // 1..100
    }
    CHECK(lob::percentile(values, 0.0) == 1);
    CHECK(lob::percentile(values, 0.50) == 50);
    CHECK(lob::percentile(values, 0.99) == 99);
    CHECK(lob::percentile(values, 0.999) == 100);
    CHECK(lob::percentile(values, 1.0) == 100);
    CHECK(lob::percentile({42}, 0.99) == 42);
}

// Empty samples and out-of-range fractions are errors.
void test_percentile_rejects_bad_input() {
    CHECK(throws<std::invalid_argument>([] { lob::percentile({}, 0.5); }));
    CHECK(throws<std::invalid_argument>([] { lob::percentile({1, 2}, 1.5); }));
}

// summarize() fills every statistic from an unsorted sample.
void test_summarize() {
    // Unsorted on purpose: summarize() must sort its own copy.
    const lob::LatencyStats stats = lob::summarize({30, 10, 20, 40});
    CHECK(stats.count == 4);
    CHECK(stats.min_ns == 10);
    CHECK(stats.max_ns == 40);
    CHECK(stats.p50_ns == 20);
    CHECK(stats.mean_ns == 25.0);
    CHECK(throws<std::invalid_argument>([] { lob::summarize({}); }));
}

// A file written in the documented layout is read back unchanged.
void test_samples_round_trip() {
    const std::string path = "test_samples.bin";
    const std::uint32_t header[4] = {lob::kSamplesMagic, 2, 3, 2};  // 2 windows of 3x2 values
    const std::vector<float> windows = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const std::vector<std::int32_t> labels = {2, 0};
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(header), sizeof(header));
        file.write(reinterpret_cast<const char*>(windows.data()),
                   static_cast<std::streamsize>(windows.size() * sizeof(float)));
        file.write(reinterpret_cast<const char*>(labels.data()),
                   static_cast<std::streamsize>(labels.size() * sizeof(std::int32_t)));
    }

    const lob::Samples samples = lob::load_samples(path);
    CHECK(samples.count == 2);
    CHECK(samples.window == 3);
    CHECK(samples.n_features == 2);
    CHECK(samples.window_size() == 6);
    CHECK(samples.window_data(1)[0] == 6.0f);
    CHECK(samples.labels == labels);
    std::remove(path.c_str());
}

// Missing and truncated files are reported, not silently accepted.
void test_samples_rejects_bad_files() {
    CHECK(throws<std::runtime_error>([] { lob::load_samples("does_not_exist.bin"); }));

    const std::string path = "test_truncated.bin";
    const std::uint32_t header[4] = {lob::kSamplesMagic, 2, 3, 2};  // promises data that is missing
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(header), sizeof(header));
    }
    CHECK(throws<std::runtime_error>([&] { lob::load_samples(path); }));
    std::remove(path.c_str());
}

// A multiplier of 2^30 with shift 31 is exactly 0.5, which exposes the rounding rule.
void test_rescale_rounds_to_nearest() {
    const std::int32_t half = 1 << 30;
    CHECK(lob::rescale(4, half, 31) == 2);
    CHECK(lob::rescale(3, half, 31) == 2);    // 1.5 rounds up
    CHECK(lob::rescale(-3, half, 31) == -1);  // -1.5 rounds towards +infinity
    CHECK(lob::rescale(-4, half, 31) == -2);
    CHECK(lob::rescale(1000, half, 30) == 1000);  // multiplier / 2^shift == 1
}

// Values outside the int8 range are clamped, not wrapped.
void test_saturate_int8() {
    CHECK(lob::saturate_int8(300) == 127);
    CHECK(lob::saturate_int8(-300) == -128);
    CHECK(lob::saturate_int8(-5) == -5);
}

// A layer small enough to compute by hand: 1 channel, kernel [1, -1], i.e. x[t] - x[t+1].
lob::FixedLayer make_difference_layer() {
    lob::FixedLayer layer;
    layer.in_channels = 1;
    layer.out_channels = 1;
    layer.kernel = 2;
    layer.stride = 1;
    layer.weights = {1, -1};
    layer.bias = {0};
    layer.multiplier = {1 << 30};  // together with shift 30: scale of exactly 1
    layer.shift = {30};
    layer.zero_point = 0;
    return layer;
}

// Convolution without ReLU reproduces the hand-computed differences.
void test_fixed_layer_convolution() {
    const lob::FixedLayer layer = make_difference_layer();
    const std::vector<std::int8_t> in = {5, 3, 10};
    std::vector<std::int8_t> out(2);

    CHECK(layer.out_length(in.size()) == 2);
    layer.forward(in.data(), in.size(), out.data());
    CHECK(out[0] == 2);   // 5 - 3
    CHECK(out[1] == -7);  // 3 - 10
}

// ReLU zeroes negative values, then the result moves to the new zero point.
void test_fixed_layer_relu() {
    lob::FixedLayer layer = make_difference_layer();
    layer.has_relu = true;
    layer.relu_multiplier = 1 << 30;
    layer.relu_shift = 30;
    layer.relu_zero_point = -128;
    const std::vector<std::int8_t> in = {5, 3, 10};
    std::vector<std::int8_t> out(2);

    layer.forward(in.data(), in.size(), out.data());
    CHECK(out[0] == -126);  // max(2, 0) - 128
    CHECK(out[1] == -128);  // max(-7, 0) - 128
}

// Strides skip input steps: with stride 2 only x[0]-x[1] and x[2]-x[3] remain.
void test_fixed_layer_stride() {
    lob::FixedLayer layer = make_difference_layer();
    layer.stride = 2;
    const std::vector<std::int8_t> in = {9, 4, 1, 8, 7};
    std::vector<std::int8_t> out(2);

    CHECK(layer.out_length(in.size()) == 2);
    layer.forward(in.data(), in.size(), out.data());
    CHECK(out[0] == 5);   // 9 - 4
    CHECK(out[1] == -7);  // 1 - 8
}

// The input stage centres prices on the latest mid, z-scores and rounds to int8.
void test_fixed_net_quantize_input() {
    lob::FixedNet net;
    net.window = 2;
    net.n_features = 4;  // [ask price, ask size, bid price, bid size]
    net.mean = {0.0f, 0.0f, 0.0f, 0.0f};
    net.std_dev = {1.0f, 1.0f, 1.0f, 1.0f};
    net.input_scale = 0.5f;
    net.input_zero_point = 0;

    // Latest snapshot has ask 11 and bid 9, so the mid is 10.
    const std::vector<float> raw = {12.0f, 3.0f, 8.0f, 1000.0f,   // older snapshot
                                    11.0f, 1.0f, 9.0f, 2.0f};     // latest snapshot
    std::vector<std::int8_t> out(8);
    net.quantize_input(raw.data(), out.data());

    CHECK(out[0] == 4);    // (12 - 10) / 0.5
    CHECK(out[1] == 6);    // sizes are not centred: 3 / 0.5
    CHECK(out[2] == -4);   // (8 - 10) / 0.5
    CHECK(out[3] == 127);  // 2000 saturates
    CHECK(out[4] == 2);    // (11 - 10) / 0.5
    CHECK(out[6] == -2);   // (9 - 10) / 0.5
}

// A missing model file is reported, not silently accepted.
void test_fixed_net_rejects_missing_file() {
    CHECK(throws<std::runtime_error>([] { lob::load_fixed_net("does_not_exist.bin"); }));
}

}  // namespace

// Runs every test; the exit code tells CTest whether all of them passed.
int main() {
    test_percentile_nearest_rank();
    test_percentile_rejects_bad_input();
    test_summarize();
    test_samples_round_trip();
    test_samples_rejects_bad_files();
    test_rescale_rounds_to_nearest();
    test_saturate_int8();
    test_fixed_layer_convolution();
    test_fixed_layer_relu();
    test_fixed_layer_stride();
    test_fixed_net_quantize_input();
    test_fixed_net_rejects_missing_file();

    if (failures == 0) {
        std::puts("all tests passed");
    }
    return failures == 0 ? 0 : 1;
}
