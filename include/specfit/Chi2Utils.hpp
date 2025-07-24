#pragma once
#include "Types.hpp"

namespace specfit {

/*  Add a *constant* error term  σ_add  (quadratically)  and return the new
 *  χ² that results:
 *
 *        σ_i²  ⟶  σ_i² + σ_add² .
 */
double chi2_with_flat_error(const Vector& resid,
                            const Vector& sigma,
                            double        sigma_add);

/*  Find σ_add  so that                                    χ² / DoF ≃ 1 .
 *  A monotonic bisection is perfectly sufficient because χ²(σ_add) is a
 *  strictly decreasing function of σ_add.
 */
double flat_error_for_reduced_chi2(const Vector& resid,
                                   const Vector& sigma,
                                   int           dof,
                                   double        tol  = 1e-4,
                                   int           it_max = 60);

} // namespace specfit