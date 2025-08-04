/* ========================================================================== */
/*  Resolution.cpp  –  modern vs. legacy spectral-resolution degradation      */
/*                                                                           */
/*  Compile normally      → modern algorithm                                 */
/*  Compile with                                                              
*      -DDIGGA_LEGACY_CONVOLUTION  → legacy algorithm (pre-upgrade behaviour) */
/* ========================================================================== */

#include "specfit/Resolution.hpp"
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace specfit {

using Vector = Eigen::VectorXd;

/* -------------------------------------------------------------------------- */
/*  Constants shared by both implementations                                  */
/* -------------------------------------------------------------------------- */
namespace {
constexpr double SIGMA_FROM_FWHM = 1.0 / (2.0 * std::sqrt(2.0 * std::log(2.0)));
constexpr double KERNEL_RADIUS   = 5.0;      // ± 5 σ
}

/* -------------------------------------------------------------------------- */
/*  1. Modern, more accurate implementation (already present before)          */
/* -------------------------------------------------------------------------- */
static Vector degrade_resolution_modern(const Vector& lam,
                                        const Vector& flux,
                                        double        resOffset,
                                        double        resSlope)
{
    const std::size_t n = lam.size();
    Vector out(n);

    /* bin width dλ ------------------------------------------------------ */
    Vector dLam(n);
    dLam[0]     = lam[1]     - lam[0];
    dLam[n - 1] = lam[n - 1] - lam[n - 2];
    for (std::size_t j = 1; j < n - 1; ++j)
        dLam[j] = 0.5 * (lam[j + 1] - lam[j - 1]);

    const double* lamData  = lam.data();
    const double* fluxData = flux.data();
    const double* dLamData = dLam.data();

    /* main loop ---------------------------------------------------------- */
    #pragma omp parallel for schedule(static) if (_OPENMP)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
    {
        const double λ_i   = lamData[i];
        const double R     = resOffset + resSlope * λ_i;
        const double sigma = (λ_i / R) * SIGMA_FROM_FWHM;

        const double lamMin = λ_i - KERNEL_RADIUS * sigma;
        const double lamMax = λ_i + KERNEL_RADIUS * sigma;

        const std::size_t jStart = std::lower_bound(lamData, lamData + n, lamMin) - lamData;
        const std::size_t jEnd   = std::upper_bound(lamData + jStart, lamData + n, lamMax) - lamData;
        const std::size_t segLen = jEnd - jStart;

        Eigen::Map<const Vector> lamSeg  (lamData  + jStart, segLen);
        Eigen::Map<const Vector> fluxSeg (fluxData + jStart, segLen);
        Eigen::Map<const Vector> dLamSeg (dLamData + jStart, segLen);

        const double inv2σ2 = 1.0 / (2.0 * sigma * sigma);

        Eigen::ArrayXd Δ      = (lamSeg.array() - λ_i);
        Eigen::ArrayXd weight = (-Δ.square() * inv2σ2).exp();
        weight *= dLamSeg.array();                       // ← modern: bin-width aware

        const double wSum   = weight.sum();
        const double fSigma = weight.matrix().dot(fluxSeg);

        out[i] = fSigma / wSum;
    }

    return out;
}

/* -------------------------------------------------------------------------- */
/*  2. Legacy implementation (pre-upgrade)                                    */
/*     Differences to the modern variant:                                     */
/*       – ignores variable bin width (treats data as evenly spaced)          */
/*       – uses a slightly coarser trapezoidal integration                    */
/* -------------------------------------------------------------------------- */
#ifdef DIGGA_LEGACY_CONVOLUTION
static Vector degrade_resolution_legacy(const Vector& lam,
                                        const Vector& flux,
                                        double        resOffset,
                                        double        resSlope)
{
    const std::size_t n = lam.size();
    Vector out(n);

    const double* lamData  = lam.data();
    const double* fluxData = flux.data();

    #pragma omp parallel for schedule(static) if (_OPENMP)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
    {
        const double λ_i   = lamData[i];
        const double R     = resOffset + resSlope * λ_i;
        const double sigma = (λ_i / R) * SIGMA_FROM_FWHM;

        const double lamMin = λ_i - KERNEL_RADIUS * sigma;
        const double lamMax = λ_i + KERNEL_RADIUS * sigma;

        const std::size_t jStart = std::lower_bound(lamData, lamData + n, lamMin) - lamData;
        const std::size_t jEnd   = std::upper_bound(lamData + jStart, lamData + n, lamMax) - lamData;
        const std::size_t segLen = jEnd - jStart;

        Eigen::Map<const Vector> lamSeg  (lamData  + jStart, segLen);
        Eigen::Map<const Vector> fluxSeg (fluxData + jStart, segLen);

        const double inv2σ2 = 1.0 / (2.0 * sigma * sigma);

        Eigen::ArrayXd Δ      = (lamSeg.array() - λ_i);
        Eigen::ArrayXd weight = (-Δ.square() * inv2σ2).exp();   // ← legacy: no dλ factor

        /* Simple trapezoidal rule (≈legacy C code) ---------------------- */
        const double wSum   = weight.sum();
        const double fSigma = weight.matrix().dot(fluxSeg);

        out[i] = fSigma / wSum;
    }

    return out;
}
#endif /* DIGGA_LEGACY_CONVOLUTION */

/* -------------------------------------------------------------------------- */
/*  Public dispatcher                                                         */
/* -------------------------------------------------------------------------- */
Vector degrade_resolution(const Vector& lam,
                          const Vector& flux,
                          double        resOffset,
                          double        resSlope)
{
#ifdef DIGGA_USE_CUDA
    return degrade_resolution_cuda(lam, flux, resOffset, resSlope);
#else
  #ifdef DIGGA_LEGACY_CONVOLUTION
    return degrade_resolution_legacy(lam, flux, resOffset, resSlope);
  #else
    return degrade_resolution_modern(lam, flux, resOffset, resSlope);
  #endif
#endif
}

} // namespace specfit