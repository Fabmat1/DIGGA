#include "specfit/Resolution.hpp"
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace specfit {

using Vector = Eigen::VectorXd;

/* ====================================================================== */
/*  CPU implementation (unchanged, only the name was changed)            */
/* ====================================================================== */
static Vector degrade_resolution_cpu(const Vector& lam,
                                     const Vector& flux,
                                     double        resOffset,
                                     double        resSlope)
{
    constexpr double SIGMA_FROM_FWHM = 1.0 / (2.0 * std::sqrt(2.0 * std::log(2.0)));
    constexpr double KERNEL_RADIUS   = 5.0;   // ±5 σ  ⇒  99.9999 % of kernel

    const std::size_t n = lam.size();
    Vector out(n);

    /* ---------------------------- dλ ---------------------------------- */
    Vector binWidth(n);
    binWidth[0]     = lam[1]     - lam[0];
    binWidth[n - 1] = lam[n - 1] - lam[n - 2];
    for (std::size_t j = 1; j < n - 1; ++j)
        binWidth[j] = 0.5 * (lam[j + 1] - lam[j - 1]);

    const double* lamData  = lam.data();
    const double* fluxData = flux.data();
    const double* dLamData = binWidth.data();

    /* --------------------------- λ_i loop ----------------------------- */
    #pragma omp parallel for schedule(static) if (_OPENMP)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
    {
        const double lambda_i = lamData[i];

        /* 1. σ(λ_i) ---------------------------------------------------- */
        const double R     = resOffset + resSlope * lambda_i;
        const double sigma = (lambda_i / R) * SIGMA_FROM_FWHM;

        /* 2. indices of ±5σ interval ---------------------------------- */
        const double lamMin = lambda_i - KERNEL_RADIUS * sigma;
        const double lamMax = lambda_i + KERNEL_RADIUS * sigma;

        const std::size_t jStart = std::lower_bound(lamData, lamData + n, lamMin) - lamData;
        const std::size_t jEnd   = std::upper_bound(lamData + jStart, lamData + n, lamMax) - lamData;

        const std::size_t segLen = jEnd - jStart;

        Eigen::Map<const Vector> lamSeg  (lamData  + jStart, segLen);
        Eigen::Map<const Vector> fluxSeg (fluxData + jStart, segLen);
        Eigen::Map<const Vector> dLamSeg (dLamData + jStart, segLen);

        /* 3. Gaussian weights ----------------------------------------- */
        const double inv2Sigma2 = 1.0 / (2.0 * sigma * sigma);

        Eigen::ArrayXd delta   = (lamSeg.array() - lambda_i);
        Eigen::ArrayXd weights = (-delta.square() * inv2Sigma2).exp();
        weights *= dLamSeg.array();

        /* 4. Weighted average ----------------------------------------- */
        const double weightSum  = weights.sum();
        const double fluxWeight = weights.matrix().dot(fluxSeg);

        out[i] = fluxWeight / weightSum;
    }

    return out;
}

/* ====================================================================== */
/*  Public dispatcher                                                     */
/* ====================================================================== */
Vector degrade_resolution(const Vector& lam,
                          const Vector& flux,
                          double        resOffset,
                          double        resSlope)
{
#ifdef SPECFIT_USE_CUDA
    return degrade_resolution_cuda(lam, flux, resOffset, resSlope);
#else
    return degrade_resolution_cpu(lam, flux, resOffset, resSlope);
#endif
}

} // namespace specfit