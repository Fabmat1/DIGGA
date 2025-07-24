#include "specfit/Chi2Utils.hpp"
#include <cmath>

namespace specfit {

double chi2_with_flat_error(const Vector& r,
                            const Vector& s,
                            double        s_add)
{
    double chi2 = 0.0;
    const double s2 = s_add * s_add;
    for (int i = 0; i < r.size(); ++i) {
        const double w = 1.0 / (s[i] * s[i] + s2);
        chi2 += r[i] * r[i] * w;
    }
    return chi2;
}

double flat_error_for_reduced_chi2(const Vector& r,
                                   const Vector& s,
                                   int           dof,
                                   double        tol,
                                   int           it_max)
{
    if (dof <= 0) return 0.0;

    /*  lower / upper bounds – start with 0 … 10·max(σ)                       */
    double lo = 0.0;
    double hi = 10.0 * s.maxCoeff();
    double chi2_lo = chi2_with_flat_error(r, s, lo);
    double chi2_hi = chi2_with_flat_error(r, s, hi);

    if (chi2_lo <= dof) return 0.0;               // already ≤ 1

    /*  enlarge upper bound until χ²(dof) < dof                               */
    while (chi2_hi > dof) {
        hi *= 2.0;
        chi2_hi = chi2_with_flat_error(r, s, hi);
    }

    /*  bisection                                                             */
    for (int it = 0; it < it_max; ++it) {
        double mid = 0.5 * (lo + hi);
        double chi2_mid = chi2_with_flat_error(r, s, mid);
        if (std::abs(chi2_mid - dof) / dof < tol) return mid;
        (chi2_mid > dof) ? lo = mid : hi = mid;
    }
    return 0.5 * (lo + hi);
}

} // namespace specfit