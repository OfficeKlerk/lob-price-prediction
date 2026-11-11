// Integer-only implementation of the quantised CNN, with no inference runtime.
//
// This is the arithmetic an FPGA would perform: int8 activations and weights,
// int32 accumulators, and rescaling by an integer multiply and a right shift.
// Floating point is used in exactly one place, the input stage, which
// normalises the raw order book and converts it to int8.
//
// Every layer is a 1-D convolution over activations stored as [time][channel];
// the dense layers of the head are convolutions whose kernel spans the whole
// time axis. The model file is written by `python -m lob.export_fixed`, which
// also documents its layout.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lob {

// First four bytes of every valid model file ("LOBF").
inline constexpr std::uint32_t kFixedMagic = 0x46424F4C;

// Number of output classes (up, stationary, down).
inline constexpr std::size_t kFixedNumClasses = 3;

// Multiplies by a real-valued scale expressed as multiplier / 2^shift,
// rounding to nearest (ties towards +infinity). In hardware this is one
// multiplier, one adder for the rounding constant, and a wire shift.
inline std::int32_t rescale(std::int32_t value, std::int32_t multiplier, std::int32_t shift) {
    const std::int64_t product = static_cast<std::int64_t>(value) * multiplier;
    const std::int64_t rounding = std::int64_t{1} << (shift - 1);
    return static_cast<std::int32_t>((product + rounding) >> shift);
}

// Clamps a value to the int8 range instead of letting it wrap around.
inline std::int8_t saturate_int8(std::int32_t value) {
    return static_cast<std::int8_t>(std::clamp<std::int32_t>(value, -128, 127));
}

// One convolution layer with its output rescaling and optional ReLU.
struct FixedLayer {
    // Shape of the convolution.
    std::size_t in_channels = 0;
    std::size_t out_channels = 0;
    std::size_t kernel = 0;
    std::size_t stride = 1;

    // Weights as [out_channel][kernel][in_channel]; the bias already includes
    // the correction for the input zero point.
    std::vector<std::int8_t> weights;
    std::vector<std::int32_t> bias;

    // Per-output-channel rescaling of the accumulator to int8.
    std::vector<std::int32_t> multiplier;
    std::vector<std::int32_t> shift;
    std::int32_t zero_point = 0;

    // Optional ReLU, followed by a rescaling to the next layer's input scale.
    bool has_relu = false;
    std::int32_t relu_multiplier = 0;
    std::int32_t relu_shift = 1;
    std::int32_t relu_zero_point = 0;

    // Length of the output time axis for an input of `in_length` steps.
    std::size_t out_length(std::size_t in_length) const {
        return (in_length - kernel) / stride + 1;
    }

    // Computes the layer. `in` holds in_length * in_channels values and `out`
    // receives out_length(in_length) * out_channels values, both as [time][channel].
    void forward(const std::int8_t* in, std::size_t in_length, std::int8_t* out) const {
        const std::size_t taps = kernel * in_channels;  // multiplications per output value
        const std::size_t steps = out_length(in_length);

        for (std::size_t t = 0; t < steps; ++t) {
            // The kernel covers `taps` consecutive input values starting here.
            const std::int8_t* x = in + t * stride * in_channels;

            for (std::size_t o = 0; o < out_channels; ++o) {
                // Multiply-accumulate in 32 bits.
                const std::int8_t* w = weights.data() + o * taps;
                std::int32_t acc = bias[o];
                for (std::size_t i = 0; i < taps; ++i) {
                    acc += static_cast<std::int32_t>(x[i]) * static_cast<std::int32_t>(w[i]);
                }

                // Rescale the accumulator to int8.
                std::int32_t q = saturate_int8(rescale(acc, multiplier[o], shift[o]) + zero_point);

                // ReLU on the real value (q - zero_point), then move to the next scale.
                if (has_relu) {
                    const std::int32_t positive = std::max(q - zero_point, 0);
                    q = saturate_int8(rescale(positive, relu_multiplier, relu_shift) +
                                      relu_zero_point);
                }
                out[t * out_channels + o] = static_cast<std::int8_t>(q);
            }
        }
    }
};

// The whole network: float input stage followed by integer layers.
class FixedNet {
public:
    // Input shape.
    std::size_t window = 0;
    std::size_t n_features = 0;

    // Input stage: normalisation constants and the float -> int8 conversion.
    std::vector<float> mean;
    std::vector<float> std_dev;
    float input_scale = 1.0f;
    std::int32_t input_zero_point = 0;

    // Integer layers, in execution order.
    std::vector<FixedLayer> layers;

    // Test vectors: ONNX Runtime's int8 logits for the benchmark samples.
    std::vector<std::int8_t> reference_logits;

    // Allocates the two activation buffers the layers alternate between.
    // Must be called once after the fields above are filled in.
    void prepare() {
        std::size_t largest = window * n_features;
        std::size_t length = window;
        for (const FixedLayer& layer : layers) {
            length = layer.out_length(length);
            largest = std::max(largest, length * layer.out_channels);
        }
        buffer_a_.assign(largest, 0);
        buffer_b_.assign(largest, 0);
    }

    // Normalises one raw window and converts it to int8.
    // `raw` holds window * n_features floats as [time][feature]. The operations
    // and their order mirror the ONNX graph so that both produce identical bits.
    void quantize_input(const float* raw, std::int8_t* out) const {
        // Mid-price of the latest snapshot: (best ask + best bid) / 2.
        const float* latest = raw + (window - 1) * n_features;
        const float mid = (latest[0] + latest[2]) / 2.0f;

        for (std::size_t t = 0; t < window; ++t) {
            for (std::size_t f = 0; f < n_features; ++f) {
                // Prices sit at even feature indices and are centred on the mid.
                const float centre = (f % 2 == 0) ? mid : 0.0f;
                const float normalised = (raw[t * n_features + f] - centre - mean[f]) / std_dev[f];
                // nearbyint rounds half to even, as ONNX QuantizeLinear does.
                const float scaled = std::nearbyint(normalised / input_scale);
                // Clamp in float first: converting an out-of-range float to int is undefined.
                const float clamped = std::clamp(scaled + static_cast<float>(input_zero_point),
                                                 -128.0f, 127.0f);
                out[t * n_features + f] = static_cast<std::int8_t>(clamped);
            }
        }
    }

    // Runs the network on one raw window and returns a pointer to its
    // kFixedNumClasses int8 logits (valid until the next call).
    const std::int8_t* predict(const float* raw) {
        std::int8_t* current = buffer_a_.data();
        std::int8_t* next = buffer_b_.data();
        quantize_input(raw, current);

        std::size_t length = window;
        for (const FixedLayer& layer : layers) {
            layer.forward(current, length, next);
            length = layer.out_length(length);
            std::swap(current, next);
        }
        return current;
    }

private:
    std::vector<std::int8_t> buffer_a_;
    std::vector<std::int8_t> buffer_b_;
};

namespace detail {

// Reads one plain value of type T from a binary stream.
template <typename T>
T read_value(std::ifstream& file) {
    T value{};
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

// Reads `count` plain values of type T from a binary stream.
template <typename T>
std::vector<T> read_array(std::ifstream& file, std::size_t count) {
    std::vector<T> values(count);
    file.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(count * sizeof(T)));
    return values;
}

}  // namespace detail

// Loads a model file; throws std::runtime_error if it is missing or malformed.
inline FixedNet load_fixed_net(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open fixed-point model: " + path);
    }
    using detail::read_array;
    using detail::read_value;

    // Header.
    if (read_value<std::uint32_t>(file) != kFixedMagic) {
        throw std::runtime_error("not a fixed-point model file: " + path);
    }
    FixedNet net;
    const auto n_layers = read_value<std::uint32_t>(file);
    net.window = read_value<std::uint32_t>(file);
    net.n_features = read_value<std::uint32_t>(file);

    // Input stage.
    net.mean = read_array<float>(file, net.n_features);
    net.std_dev = read_array<float>(file, net.n_features);
    net.input_scale = read_value<float>(file);
    net.input_zero_point = read_value<std::int32_t>(file);

    // Layers.
    for (std::uint32_t i = 0; i < n_layers; ++i) {
        FixedLayer layer;
        layer.in_channels = read_value<std::uint32_t>(file);
        layer.out_channels = read_value<std::uint32_t>(file);
        layer.kernel = read_value<std::uint32_t>(file);
        layer.stride = read_value<std::uint32_t>(file);
        layer.has_relu = read_value<std::uint32_t>(file) != 0;

        layer.weights = read_array<std::int8_t>(
            file, layer.out_channels * layer.kernel * layer.in_channels);
        layer.bias = read_array<std::int32_t>(file, layer.out_channels);
        layer.multiplier = read_array<std::int32_t>(file, layer.out_channels);
        layer.shift = read_array<std::int32_t>(file, layer.out_channels);
        layer.zero_point = read_value<std::int32_t>(file);
        if (layer.has_relu) {
            layer.relu_multiplier = read_value<std::int32_t>(file);
            layer.relu_shift = read_value<std::int32_t>(file);
            layer.relu_zero_point = read_value<std::int32_t>(file);
        }
        net.layers.push_back(std::move(layer));
    }

    // Test vectors.
    const auto n_reference = read_value<std::uint32_t>(file);
    net.reference_logits = read_array<std::int8_t>(file, n_reference * kFixedNumClasses);

    if (!file) {
        throw std::runtime_error("fixed-point model file is truncated: " + path);
    }
    net.prepare();
    return net;
}

}  // namespace lob
