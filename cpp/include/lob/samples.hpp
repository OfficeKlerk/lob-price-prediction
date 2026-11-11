// Loader for the benchmark sample file written by `python -m lob.export`.
//
// File layout (little-endian):
//     uint32  magic, n_samples, window, n_features
//     float32 windows[n_samples * window * n_features]
//     int32   labels[n_samples]
#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lob {

// First four bytes of every valid samples file.
inline constexpr std::uint32_t kSamplesMagic = 0x53424F4C;  // "LOBS"

// Benchmark inputs held in memory: test windows and their true labels.
struct Samples {
    std::size_t count = 0;       // number of windows
    std::size_t window = 0;      // snapshots per window
    std::size_t n_features = 0;  // values per snapshot
    std::vector<float> windows;  // count * window * n_features values, row-major
    std::vector<std::int32_t> labels;

    // Number of float values in one window.
    std::size_t window_size() const { return window * n_features; }

    // Pointer to the first value of window `i`.
    const float* window_data(std::size_t i) const { return windows.data() + i * window_size(); }
};

// Reads a whole samples file; throws std::runtime_error if it is missing,
// malformed or truncated.
inline Samples load_samples(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open samples file: " + path);
    }

    // Header: magic number followed by the three dimensions.
    std::uint32_t header[4] = {};
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!file || header[0] != kSamplesMagic) {
        throw std::runtime_error("not a samples file (bad header): " + path);
    }

    // Allocate the buffers the header promises.
    Samples samples;
    samples.count = header[1];
    samples.window = header[2];
    samples.n_features = header[3];
    samples.windows.resize(samples.count * samples.window_size());
    samples.labels.resize(samples.count);

    // Payload: all windows first, then all labels.
    file.read(reinterpret_cast<char*>(samples.windows.data()),
              static_cast<std::streamsize>(samples.windows.size() * sizeof(float)));
    file.read(reinterpret_cast<char*>(samples.labels.data()),
              static_cast<std::streamsize>(samples.labels.size() * sizeof(std::int32_t)));
    if (!file) {
        throw std::runtime_error("samples file is truncated: " + path);
    }
    return samples;
}

}  // namespace lob
