#pragma once

#include "Types.hpp"
#include <boost/math/interpolators/makima.hpp>

namespace specfit {

class AkimaSpline {
private:
    // Your underlying spline implementation (e.g., boost::math::interpolators::makima)
    // Adjust this type based on your actual spline library
    mutable decltype(boost::math::interpolators::makima(
        std::vector<Real>(), std::vector<Real>())) spline_;
    
    // Boundary information for extrapolation
    Real x_min_, x_max_;
    Real y_min_, y_max_;
    Real deriv_min_, deriv_max_;

public:
    AkimaSpline(const Vector& x, const Vector& y);
    
    Real operator()(Real x) const;
    Vector operator()(const Vector& x) const;
};

} // namespace specfit