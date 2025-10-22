// RotationalConvolution.cpp
#include "specfit/RotationalConvolution.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

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

/* ------------- Linear interpolation helper --------------------------- */
static double linear_interp(const Vector& x, const Vector& y, double xi)
{
    const std::ptrdiff_t n = x.size();
    if (n == 0) return 0.0;
    if (n == 1) return y[0];
    if (xi <= x[0]) return y[0];
    if (xi >= x[n-1]) return y[n-1];
    
    auto it = std::lower_bound(x.begin(), x.end(), xi);
    std::ptrdiff_t i = std::distance(x.begin(), it);
    if (i == 0) i = 1;
    if (i >= n) i = n - 1;
    
    const double t = (xi - x[i-1]) / (x[i] - x[i-1]);
    return y[i-1] * (1.0 - t) + y[i] * t;
}

/* ------------- Kernel computation ------------------------------------ */
static RotationalKernelCache::KernelPtr 
compute_kernel(double vsini_kms, double epsilon, int n_kernel)
{
    auto kernel = std::make_shared<RotationalKernelCache::KernelData>();
    kernel->vsini_kms = vsini_kms;
    kernel->epsilon = epsilon;
    kernel->n_kernel = n_kernel;
    
    if (n_kernel <= 0) n_kernel = 81;
    if (n_kernel % 2 == 0) n_kernel++;
    
    kernel->vel_shift.resize(n_kernel);
    kernel->vel_kernel.resize(n_kernel);
    
    // Create velocity grid from -vsini to +vsini
    for (int i = 0; i < n_kernel; ++i) {
        double x = -1.0 + 2.0 * i / (n_kernel - 1);  // -1 to +1
        kernel->vel_shift[i] = x * vsini_kms;  // Convert to km/s
        kernel->vel_kernel[i] = rot_profile(x, epsilon);
    }
    
    // Normalize the kernel
    double kernel_sum = kernel->vel_kernel.sum();
    if (kernel_sum > 0) {
        kernel->vel_kernel /= kernel_sum;
    }
    
    return kernel;
}

/* ------------- Cache implementation ---------------------------------- */
RotationalKernelCache::KernelPtr 
RotationalKernelCache::get_or_compute(double vsini_kms, double epsilon, int n_kernel)
{
    if (vsini_kms <= 0.0) {
        // Return delta function kernel for zero rotation
        auto kernel = std::make_shared<KernelData>();
        kernel->vsini_kms = 0.0;
        kernel->epsilon = epsilon;
        kernel->n_kernel = 1;
        kernel->vel_shift = Vector::Zero(1);
        kernel->vel_kernel = Vector::Ones(1);
        return kernel;
    }
    
    Key key{vsini_kms, epsilon, n_kernel};
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;  // Cache hit
        }
    }
    
    // Compute outside of lock
    auto kernel = compute_kernel(vsini_kms, epsilon, n_kernel);
    
    // Store in cache
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Check again in case another thread computed it
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
        
        // Limit cache size (optional)
        if (cache_.size() > 100) {  // Keep last 100 kernels
            cache_.clear();  // Simple strategy; could use LRU instead
        }
        
        cache_[key] = kernel;
    }
    
    return kernel;
}

/* ------------------- Main interface with caching --------------------- */
Vector rotational_broaden(const Vector& lam,
                         const Vector& flux,
                         double vsini_kms,
                         double epsilon,
                         int n_kernel)
{
    const std::ptrdiff_t N = flux.size();
    if (N == 0 || vsini_kms <= 0.0) return flux;
    
    // Get cached kernel
    auto kernel = RotationalKernelCache::instance()
                    .get_or_compute(vsini_kms, epsilon, n_kernel);
    
    const double c_km = 299792.458;  // Speed of light in km/s
    Vector out(N);
    
    // For each wavelength point
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < N; ++i) {
        double lam_center = lam[i];
        double sum_weight = 0.0;
        double sum_flux = 0.0;
        
        // Convert velocity shifts to wavelength shifts at this wavelength
        for (int k = 0; k < kernel->n_kernel; ++k) {
            double dlam = lam_center * kernel->vel_shift[k] / c_km;
            double lam_k = lam_center + dlam;
            
            // Find flux at this wavelength via interpolation
            if (lam_k >= lam[0] && lam_k <= lam[N-1]) {
                double flux_k = linear_interp(lam, flux, lam_k);
                sum_flux += flux_k * kernel->vel_kernel[k];
                sum_weight += kernel->vel_kernel[k];
            }
        }
        
        out[i] = (sum_weight > 0) ? sum_flux / sum_weight : flux[i];
    }
    
    return out;
}

/* ------------------- Version without caching (for testing) ----------- */
Vector rotational_broaden_nocache(const Vector& lam,
                                  const Vector& flux,
                                  double vsini_kms,
                                  double epsilon,
                                  int n_kernel)
{
    const std::ptrdiff_t N = flux.size();
    if (N == 0 || vsini_kms <= 0.0) return flux;
    
    auto kernel = compute_kernel(vsini_kms, epsilon, n_kernel);
    
    // ... rest is same as cached version ...
    const double c_km = 299792.458;
    Vector out(N);
    
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < N; ++i) {
        double lam_center = lam[i];
        double sum_weight = 0.0;
        double sum_flux = 0.0;
        
        for (int k = 0; k < kernel->n_kernel; ++k) {
            double dlam = lam_center * kernel->vel_shift[k] / c_km;
            double lam_k = lam_center + dlam;
            
            if (lam_k >= lam[0] && lam_k <= lam[N-1]) {
                double flux_k = linear_interp(lam, flux, lam_k);
                sum_flux += flux_k * kernel->vel_kernel[k];
                sum_weight += kernel->vel_kernel[k];
            }
        }
        
        out[i] = (sum_weight > 0) ? sum_flux / sum_weight : flux[i];
    }
    
    return out;
}

} // namespace specfit