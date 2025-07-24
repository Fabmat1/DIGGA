#pragma once
#include "Types.hpp"

namespace specfit {

Vector rotational_broaden(const Vector& lam,
                          const Vector& flux,
                          double vsini_kms,
                          double epsilon = 0.6); // linear limb darkening

} // namespace specfit