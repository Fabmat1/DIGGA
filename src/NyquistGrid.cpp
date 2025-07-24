#include "specfit/NyquistGrid.hpp"
#include <vector>

namespace specfit {

Vector build_nyquist_grid(double lambda_min,
                          double lambda_max,
                          double res_offset,
                          double res_slope,
                          double nq_eff)
{
    std::vector<Real> grid;
    double lam = lambda_min;
    const double eps = std::numeric_limits<Real>::epsilon();
    
    while (lam < lambda_max) {
        grid.push_back(lam);
    
        double R    = res_offset + res_slope * lam;
        double dlam = lam / (nq_eff * R);
    
        /* avoid duplicates: make sure the next λ is strictly larger    */
        double next = lam + dlam;
        if (next <= lam)   // step too small – push to next representable double
            next = std::nextafter(lam, std::numeric_limits<Real>::infinity());
    
        lam = next;
    }
    grid.push_back(lambda_max);
    /* remove any accidental duplicates that may still be present       */
    grid.erase(std::unique(grid.begin(), grid.end(),
                           [&](Real a, Real b) { return std::abs(a - b) <= eps * a; }),
               grid.end());
    return Eigen::Map<Vector>(grid.data(), grid.size());
}

} // namespace specfit