#pragma once
#include "Types.hpp"

namespace specfit {

Vector trapezoidal_rebin(const Vector& lam_in,
                         const Vector& flux_in,
                         const Vector& lam_out);

} // namespace specfit