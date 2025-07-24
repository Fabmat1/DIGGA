#pragma once
#include "Types.hpp"
#include "Spectrum.hpp"
#include <vector>
#include <tuple>

namespace specfit {

/**
 * Build continuum–spline anchor positions from the triples that appear in the
 * JSON configuration:
 *
 *     [ start , end , step ]
 *
 * For every triple the function returns      start , start+step , … , end
 * (end is enforced to be contained ‑ no duplicates).
 */
Vector anchors_from_intervals(
        const std::vector<std::tuple<double, double, double>>& intervals,
        const Spectrum& spectrum);

/**
 * Helper – simple linear interpolation y(x_out)  (no extrapolation).
 */
Vector interp_linear(const Vector& x_in,
                     const Vector& y_in,
                     const Vector& x_out);

} // namespace specfit
