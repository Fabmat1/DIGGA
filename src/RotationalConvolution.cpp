#include "specfit/RotationalConvolution.hpp"
#include <vector>
#include <cmath>

namespace specfit {

static double rotational_profile(double x, double eps)
{
    const double t = 1.0 - x * x;
    return t > 0 ? (2 * (1 - eps) * std::sqrt(t) + M_PI * eps * t) : 0.0;
}

// src/RotationalConvolution.cpp
Vector rotational_broaden(const Vector& lam,
                          const Vector& flux,
                          double vsini_kms,
                          double epsilon)
{
    if (vsini_kms <= 0) return flux;

    /* ----------- determine a representative wavelength step ------------ */
    double mean_dlam = 0.0;
    int    n_dlam    = 0;
    for (int i = 1; i < lam.size(); ++i) {
        const double d = lam[i] - lam[i - 1];
        if (d > 0.0) { mean_dlam += d; ++n_dlam; }
    }
    if (n_dlam == 0)            // every λ identical → nothing to do
        return flux;
    mean_dlam /= n_dlam;

    /* ------------------- build rotational kernel ----------------------- */
    const double c_km = 299'792.458;
    const double dlam = lam.mean() * vsini_kms / c_km;
    int n = static_cast<int>(std::ceil(10.0 * dlam / mean_dlam));
    if (n < 3) n = 3;
    if (n % 2 == 0) ++n;                        // enforce odd length

    Vector kernel(n);
    const int mid = n / 2;
    for (int i = 0; i < n; ++i) {
        double x = (i - mid) / static_cast<double>(mid);
        kernel[i] = rotational_profile(x, epsilon);
    }
    const double norm = kernel.sum();
    if (norm == 0.0) return flux;               // pathological but safe
    kernel /= norm;

    /* -------------------- convolve ------------------------------------- */
    Vector out(flux.size());
    for (int i = 0; i < flux.size(); ++i) {
        double acc = 0.0;
        for (int k = 0; k < n; ++k) {
            int j = i + k - mid;
            if (j >= 0 && j < flux.size())
                acc += flux[j] * kernel[k];
        }
        out[i] = acc;
    }
    return out;
}

} // namespace specfit