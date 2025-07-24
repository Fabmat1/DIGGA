#include "specfit/ContinuumModel.hpp"

namespace specfit {

ContinuumModel::ContinuumModel(const Vector& anchors_x,
                               const Vector& anchors_y)
    : x_(anchors_x), y_(anchors_y), spline_(x_, y_)
{}

Vector ContinuumModel::evaluate(const Vector& lambda) const
{
    return spline_(lambda); 
}

void ContinuumModel::update_y(const Vector& new_y)
{
    y_ = new_y;
    rebuild();
}

void ContinuumModel::rebuild()
{
    spline_ = AkimaSpline(x_, y_);
}

} // namespace specfit