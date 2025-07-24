// Disable Eigen's CUDA code path – we only need Eigen on the host:
#define EIGEN_NO_CUDA

#include "specfit/Resolution.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
namespace specfit {

/* ------------------------------------------------------------------ */
/*  Device-side helpers: binary search on sorted array (lower / upper)*/
/* ------------------------------------------------------------------ */
__device__ std::size_t lower_bound_dev(const double* a,
                                       std::size_t   n,
                                       double        value)
{
    std::size_t first = 0;
    while (n) {
        std::size_t half = n >> 1;
        std::size_t mid  = first + half;
        if (a[mid] < value) {
            first = mid + 1;
            n    -= half + 1;
        } else {
            n = half;
        }
    }
    return first;
}
__device__ std::size_t upper_bound_dev(const double* a,
                                       std::size_t   n,
                                       double        value)
{
    std::size_t first = 0;
    while (n) {
        std::size_t half = n >> 1;
        std::size_t mid  = first + half;
        if (value >= a[mid]) {
            first = mid + 1;
            n    -= half + 1;
        } else {
            n = half;
        }
    }
    return first;
}

/* ------------------------------------------------------------------ */
/*  Kernel: one thread → one output λ_i                               */
/* ------------------------------------------------------------------ */
__global__ void degrade_kernel(const double* lam,
                               const double* flux,
                               const double* dLam,
                               double        resOffset,
                               double        resSlope,
                               std::size_t   n,
                               double*       out)
{
    /* pre-computed constant: 1/(2·sqrt(2·ln2)) */
    const double SIGMA_FROM_FWHM = 0.42466090014400953;
    const double KERNEL_RADIUS   = 5.0;

    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double lambda_i = lam[i];
    const double R        = resOffset + resSlope * lambda_i;
    const double sigma    = (lambda_i / R) * SIGMA_FROM_FWHM;

    const double lamMin = lambda_i - KERNEL_RADIUS * sigma;
    const double lamMax = lambda_i + KERNEL_RADIUS * sigma;

    const std::size_t jStart = lower_bound_dev(lam, n, lamMin);
    const std::size_t jEnd   = upper_bound_dev(lam, n, lamMax);

    const double inv2Sigma2 = 1.0 / (2.0 * sigma * sigma);

    double weightSum  = 0.0;
    double fluxWeight = 0.0;

    for (std::size_t j = jStart; j < jEnd; ++j) {
        const double delta  = lam[j] - lambda_i;
        const double weight = exp(-delta * delta * inv2Sigma2) * dLam[j];
        weightSum  += weight;
        fluxWeight += weight * flux[j];
    }
    out[i] = fluxWeight / weightSum;
}

/* ------------------------------------------------------------------ */
/*  Host-side wrapper                                                 */
/* ------------------------------------------------------------------ */
Vector degrade_resolution_cuda(const Vector& lam,
                               const Vector& flux,
                               double        resOffset,
                               double        resSlope)
{
    const std::size_t n = lam.size();

    /* ------------- dλ on host (same as CPU path) -------------------- */
    Vector binWidth(n);
    binWidth[0]     = lam[1]     - lam[0];
    binWidth[n - 1] = lam[n - 1] - lam[n - 2];
    for (std::size_t j = 1; j < n - 1; ++j)
        binWidth[j] = 0.5 * (lam[j + 1] - lam[j - 1]);

    /* ------------- copy to GPU -------------------------------------- */
    thrust::device_vector<double> d_lam  (lam.begin(),      lam.end());
    thrust::device_vector<double> d_flux (flux.begin(),     flux.end());
    thrust::device_vector<double> d_dLam (binWidth.begin(), binWidth.end());
    thrust::device_vector<double> d_out  (n);

    const int blockSize = 256;
    const int blocks    = static_cast<int>((n + blockSize - 1) / blockSize);

    degrade_kernel<<<blocks, blockSize>>>(
        thrust::raw_pointer_cast(d_lam.data()),
        thrust::raw_pointer_cast(d_flux.data()),
        thrust::raw_pointer_cast(d_dLam.data()),
        resOffset,
        resSlope,
        n,
        thrust::raw_pointer_cast(d_out.data())
    );
    cudaDeviceSynchronize();

    /* ------------- back to host ------------------------------------- */
    Vector out(n);
    thrust::copy(d_out.begin(), d_out.end(), out.data());
    return out;
}

} // namespace specfit