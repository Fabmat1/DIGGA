#pragma once
#include "Spectrum.hpp"
#include "ContinuumModel.hpp"
#include "Types.hpp"
#include <vector>
#include <tuple>

namespace specfit {

struct IgnoreInterval { double lo, hi; };

class Observation {
public:
    Observation(Spectrum s, std::vector<IgnoreInterval> ignore,
                ContinuumModel cont);

    const Spectrum& data() const;
    const ContinuumModel& continuum() const;

private:
    Spectrum data_;
    std::vector<IgnoreInterval> ignore_;
    ContinuumModel continuum_;
};

} // namespace specfit