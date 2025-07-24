#include "specfit/AkimaSpline.hpp"

namespace specfit {

AkimaSpline::AkimaSpline(const Vector& x, const Vector& y)
    : spline_(std::vector<Real>(x.data(), x.data() + x.size()),
              std::vector<Real>(y.data(), y.data() + y.size())),
      x_min_(x.minCoeff()),
      x_max_(x.maxCoeff())
{
    // Store boundary values and derivatives for extrapolation
    y_min_ = spline_(x_min_);
    y_max_ = spline_(x_max_);
    
    // Calculate derivatives at boundaries for linear extrapolation
    Real h = 1e-6; // Small step for numerical derivative
    deriv_min_ = (spline_(x_min_ + h) - y_min_) / h;
    deriv_max_ = (y_max_ - spline_(x_max_ - h)) / h;
}

Real AkimaSpline::operator()(Real x) const
{
    if (x < x_min_) {
        // Linear extrapolation below minimum
        return y_min_ + deriv_min_ * (x - x_min_);
    } else if (x > x_max_) {
        // Linear extrapolation above maximum
        return y_max_ + deriv_max_ * (x - x_max_);
    } else {
        // Within spline domain - use normal evaluation
        return spline_(x);
    }
}

Vector AkimaSpline::operator()(const Vector& x) const
{
    Vector out(x.size());
    for (int i = 0; i < x.size(); ++i) {
        out[i] = operator()(x[i]);
    }
    return out;
}

} // namespace specfit