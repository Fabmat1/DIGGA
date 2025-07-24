#pragma once
#include "Spectrum.hpp"
#include "ModelGrid.hpp"
#include "ContinuumModel.hpp"

namespace specfit {

struct StellarParams {
    double vrad;
    double vsini;
    double zeta;
    double teff;
    double logg;
    double xi;
    double z;
    double he;
};

Spectrum compute_synthetic(const ModelGrid& grid,
                           const StellarParams& pars,
                           const Vector& lambda_obs,
                           double resOffset,
                           double resSlope);

} // namespace specfit