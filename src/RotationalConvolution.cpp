#include "specfit/RotationalConvolution.hpp"
#include <vector>
#include <cmath>
#include <omp.h>

namespace specfit {

/* ------------- Gray (2005) rotational profile ------------------------ */
static inline double rot_profile(double x, double eps)
{
    if (std::abs(x) >= 1.0) return 0.0;
    const double t   = 1.0 - x*x;
    const double num = 2.0*(1.0-eps)*std::sqrt(t) + 0.5*M_PI*eps*t;
    const double den = M_PI*(1.0 - eps/3.0);
    return num / den;
}

/* ------------------- main interface ---------------------------------- */
Vector rotational_broaden(const Vector& lam,
                          const Vector& flux,
                          double        vsini_kms,
                          double        epsilon)
{
    /* ---------- trivial cases ---------------------------------------- */
    const std::ptrdiff_t N = flux.size();
    if (N == 0 || vsini_kms <= 0.0) return flux;

    /* ---------- mean Δλ ---------------------------------------------- */
    double dlam_mean = 0.0; std::ptrdiff_t cnt = 0;
    for (std::ptrdiff_t i = 1; i < N; ++i)
        if (lam[i] > lam[i-1]) { dlam_mean += lam[i]-lam[i-1]; ++cnt; }
    if (cnt == 0) return flux;        // identical λ → nothing to do
    dlam_mean /= cnt;

    /* ---------- kernel ------------------------------------------------ */
    const double c_km     = 299'792.458;
    const double dl_max   = lam.mean() * vsini_kms / c_km;   // half-width
    int   mid             = static_cast<int>(std::ceil(dl_max / dlam_mean));
    mid  = std::max(1, mid);
    int   klen            = 2*mid + 1;

    Vector kernel(klen);
    for (int k = 0; k < klen; ++k) {
        double dl = (k - mid) * dlam_mean;   // offset in Å
        kernel[k] = rot_profile(dl / dl_max, epsilon);
    }
    kernel /= kernel.sum();                 // normalise once!

    /* ---------- convolution via sliding dot product ------------------ */
    Vector out(N);
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < N; ++i) {

        // compute slice limits in index space
        const std::ptrdiff_t j_lo = std::max<std::ptrdiff_t>(0, i - mid);
        const std::ptrdiff_t j_hi = std::min<std::ptrdiff_t>(N - 1, i + mid);
        const std::ptrdiff_t len  = j_hi - j_lo + 1;

        // kernel slice that overlaps with the data slice
        const std::ptrdiff_t k_lo = mid - (i - j_lo);

        // Eigen vectorised dot product
        out[i] = flux.segment(j_lo, len).dot(kernel.segment(k_lo, len));
    }
    return out;
}


} // namespace specfit