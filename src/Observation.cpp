#include "specfit/Observation.hpp"

namespace specfit {

Observation::Observation(Spectrum s, std::vector<IgnoreInterval> ignore,
                         ContinuumModel cont)
    : data_(std::move(s)), ignore_(std::move(ignore)), continuum_(std::move(cont))
{}

const Spectrum& Observation::data() const { return data_; }
const ContinuumModel& Observation::continuum() const { return continuum_; }

} // namespace specfit