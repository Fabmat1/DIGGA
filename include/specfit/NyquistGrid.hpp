#pragma once
#include "Types.hpp"

namespace specfit {

Vector build_nyquist_grid(double lambda_min,
                          double lambda_max,
                          double res_offset,
                          double res_slope,
                          double nq_eff = 2.7);

} // namespace specfit