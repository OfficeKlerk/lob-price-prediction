// Summary statistics for a set of latency measurements.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace lob {

// Summary of one benchmark run; all times are in nanoseconds.
struct LatencyStats {
    std::size_t count = 0;
    double mean_ns = 0.0;
    std::int64_t min_ns = 0;
    std::int64_t p50_ns = 0;
    std::int64_t p90_ns = 0;
    std::int64_t p99_ns = 0;
    std::int64_t p999_ns = 0;
    std::int64_t max_ns = 0;
};

// Nearest-rank percentile of an ascending-sorted, non-empty vector.
// `q` is a fraction in [0, 1]; the result is always one of the measured
// values, never an interpolation between two of them.
inline std::int64_t percentile(const std::vector<std::int64_t>& sorted, double q) {
    if (sorted.empty()) {
        throw std::invalid_argument("percentile of an empty sample");
    }
    if (q < 0.0 || q > 1.0) {
        throw std::invalid_argument("percentile fraction must be in [0, 1]");
    }
    const auto rank = static_cast<std::size_t>(std::ceil(q * static_cast<double>(sorted.size())));
    return sorted[std::max<std::size_t>(rank, 1) - 1];
}

// Sorts the measurements and computes every field of LatencyStats.
// Takes the samples by value because it needs its own copy to sort.
inline LatencyStats summarize(std::vector<std::int64_t> samples_ns) {
    if (samples_ns.empty()) {
        throw std::invalid_argument("cannot summarize an empty sample");
    }
    std::sort(samples_ns.begin(), samples_ns.end());

    LatencyStats stats;
    stats.count = samples_ns.size();
    stats.mean_ns = std::accumulate(samples_ns.begin(), samples_ns.end(), 0.0) /
                    static_cast<double>(samples_ns.size());
    stats.min_ns = samples_ns.front();
    stats.p50_ns = percentile(samples_ns, 0.50);
    stats.p90_ns = percentile(samples_ns, 0.90);
    stats.p99_ns = percentile(samples_ns, 0.99);
    stats.p999_ns = percentile(samples_ns, 0.999);
    stats.max_ns = samples_ns.back();
    return stats;
}

}  // namespace lob
