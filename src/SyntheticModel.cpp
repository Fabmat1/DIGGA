#include "specfit/SyntheticModel.hpp"
#include "specfit/Resolution.hpp"            // still needed for header only
#include "specfit/ContinuumUtils.hpp"
#include "specfit/RotationalConvolution.hpp"
#include <cmath>
#include <cassert>

namespace specfit {

Spectrum compute_synthetic(const ModelGrid&    grid,
                           const StellarParams& pars,
                           const Vector&        lambda_obs,
                           double               resOffset,
                           double               resSlope)
{
    /* spectrum already has requested instrumental resolution */
    Spectrum surf = grid.load_spectrum(pars.teff, pars.logg,
                                       pars.z, pars.he, pars.xi,
                                       resOffset, resSlope);

    /* 1 – rotation --------------------------------------------------- */
    Vector rot = rotational_broaden(surf.lambda, surf.flux, pars.vsini);

    /* 2 – Doppler shift --------------------------------------------- */
    constexpr double c = 299'792.458;                 // km/s
    double factor      = 1.0 + pars.vrad / c;         // λ_obs = λ_rest * factor
    Vector lam_shift   = surf.lambda * factor;

    /* 3 – interpolate on to observed grid --------------------------- */
    Vector interp = interp_linear(lam_shift, rot, lambda_obs);

    Spectrum out;
    out.lambda = lambda_obs;
    out.flux   = interp;
    out.sigma  = Vector::Ones(lambda_obs.size());
    return out;
}

} // namespace specfit