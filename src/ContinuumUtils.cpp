#include "specfit/ContinuumUtils.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>       // std::abs

namespace specfit {

Vector anchors_from_intervals(
        const std::vector<std::tuple<double, double, double>>& intervals,
        const Spectrum& spectrum) 
{
    std::vector<Real> xs;

    // Collect non-ignored lambda values
    std::vector<double> non_ignored_lambdas;
    for (int i = 0; i < spectrum.lambda.size(); ++i) {
        if (spectrum.ignoreflag[i] == 1) {  // Non-ignored
            non_ignored_lambdas.push_back(spectrum.lambda[i]);
        }
    }

    // Sort the non-ignored lambdas to ensure proper indexing
    std::sort(non_ignored_lambdas.begin(), non_ignored_lambdas.end());

    // Determine the min and max of non-ignored lambdas
    double min_lambda = non_ignored_lambdas.empty() ? 
        std::numeric_limits<double>::max() : non_ignored_lambdas.front();
    double max_lambda = non_ignored_lambdas.empty() ? 
        std::numeric_limits<double>::lowest() : non_ignored_lambdas.back();

    // Add boundary anchor points if we have enough non-ignored lambda values
    if (non_ignored_lambdas.size() >= 20) {  // Need at least 20 points for 10th and (n-10)th
        // Add anchor at 10th index (0-based, so index 9)
        xs.push_back(non_ignored_lambdas[9]);
        // Add anchor at (length-10)th index (0-based, so index length-10)
        xs.push_back(non_ignored_lambdas[non_ignored_lambdas.size() - 10]);
    } else if (non_ignored_lambdas.size() > 0) {
        // If we don't have enough points, add the boundaries
        xs.push_back(min_lambda);
        xs.push_back(max_lambda);
    }

    for (const auto& [lo, hi, step] : intervals) {
        if (hi < lo || step <= 0) continue;
        for (double x = lo; x <= hi + 1e-6; x += step) {
            if (x >= min_lambda && x <= max_lambda) {  // Check if x is within non-ignored range
                xs.push_back(x);
            }
        }
        // Ensure 'hi' is present if it's within the non-ignored range
        if (!xs.empty() && xs.back() < hi - 1e-6 && hi >= min_lambda && hi <= max_lambda) {
            xs.push_back(hi);
        }
    }

    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());

    return Eigen::Map<const Vector>(xs.data(), xs.size());
}

/* ------------------------------------------------------------------ */

Vector interp_linear(const Vector& x_in,
                     const Vector& y_in,
                     const Vector& x_out)
{
    const Eigen::Index n_in  = x_in.size();
    const Eigen::Index n_out = x_out.size();

    Vector out(n_out);

    // -------- 1. Pre-compute slopes -----------------------------------------
    Vector slope(n_in - 1);
    for (Eigen::Index i = 0; i < n_in - 1; ++i) {
        const double dx = x_in[i+1] - x_in[i];
        slope[i] = (std::abs(dx) < 1e-12) ? 0.0
                                          : (y_in[i+1] - y_in[i]) / dx;
    }

    // -------- 2. Single forward scan through the already-sorted x_out -------
    Eigen::Index seg = 0;                 // left border of current interval

    Eigen::Index k = 0;
    // left tail
    while (k < n_out && x_out[k] <= x_in[0])
        out[k++] = y_in[0];

    // interior region
    for (; k < n_out; ++k) {
        const double x = x_out[k];

        // right tail encountered → fill remainder and stop
        if (x >= x_in[n_in-1]) {
            for (; k < n_out; ++k) out[k] = y_in[n_in-1];
            break;
        }

        // advance segment until x_in[seg] ≤ x < x_in[seg+1]
        while (x_in[seg+1] < x) ++seg;

        out[k] = y_in[seg] + slope[seg] * (x - x_in[seg]);
    }
    return out;
}

} // namespace specfit