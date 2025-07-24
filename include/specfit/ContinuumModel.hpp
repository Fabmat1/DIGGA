#pragma once
#include "AkimaSpline.hpp"
#include "Types.hpp"
#include <vector>

namespace specfit {

class ContinuumModel {
public:
    ContinuumModel(const Vector& anchors_x, const Vector& anchors_y);

    Vector evaluate(const Vector& lambda) const;   // ← multiplicative factor
    void update_y(const Vector& new_y);

private:
    Vector x_, y_;
    AkimaSpline spline_;
    void rebuild();
};

} // namespace specfit