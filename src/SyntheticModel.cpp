/* ===================================================================== *
 *  src/SyntheticModel.cpp      –– two-level ultra-fast caching
 * ===================================================================== */

#include "specfit/SyntheticModel.hpp"
#include "specfit/SpectrumCache.hpp"
#include "specfit/Resolution.hpp"
#include "specfit/ContinuumUtils.hpp"

#include <ankerl/unordered_dense.h>   // hash mixing constant
#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>
#include <iostream>

namespace specfit {

/* ------------------------------------------------------------------ *
 *  Tiny helper  –– boost-like hash_combine                           *
 * ------------------------------------------------------------------ */
namespace {

template<typename T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    seed ^= std::hash<T>{}(v) +
            0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

/* Hash for the FINAL synthetic spectrum (everything) */
inline std::size_t make_hash_full(const StellarParams& p,
                                  double               lam_min,
                                  double               lam_max,
                                  std::size_t          lam_sz,
                                  double               resOffset,
                                  double               resSlope)
{
    std::size_t seed = 0xF00DBAA5FULL;  // domain separator

    hash_combine(seed, p.vrad);
    hash_combine(seed, p.vsini);
    hash_combine(seed, p.zeta);
    hash_combine(seed, p.teff);
    hash_combine(seed, p.logg);
    hash_combine(seed, p.xi);
    hash_combine(seed, p.z);
    hash_combine(seed, p.he);

    hash_combine(seed, lam_min);
    hash_combine(seed, lam_max);
    hash_combine(seed, lam_sz);

    hash_combine(seed, resOffset);
    hash_combine(seed, resSlope);

    return seed;
}

/* Hash for the *surface* spectrum that comes from grid.load_spectrum(...)
 * (includes rotation and resolution degradation)
 */
inline std::size_t make_hash_surface(const StellarParams& p,
                                     double               resOffset,
                                     double               resSlope)
{
    std::size_t seed = 0xBADA555EULL;   // different domain separator

    hash_combine(seed, p.teff);
    hash_combine(seed, p.logg);
    hash_combine(seed, p.z);
    hash_combine(seed, p.he);
    hash_combine(seed, p.xi);
    hash_combine(seed, p.vsini);    // Now included in surface hash

    hash_combine(seed, resOffset);
    hash_combine(seed, resSlope);

    return seed;
}

} // unnamed namespace

/* ------------------------------------------------------------------ *
 *                         main routine                               *
 * ------------------------------------------------------------------ */
Spectrum compute_synthetic(const ModelGrid&    grid,
                           const StellarParams& pars,
                           const Vector&        lambda_obs,
                           double               resOffset,
                           double               resSlope)
{
    //std::cout << "[CompSynth] Entering Function" << std::endl;  
    //std::cout << "[CompSynth] Making Hash." << std::endl;               
    /* ------------------------- FULL key ---------------------------- */
    const double      lam_min  = lambda_obs.minCoeff();
    const double      lam_max  = lambda_obs.maxCoeff();
    const std::size_t lam_size = static_cast<std::size_t>(lambda_obs.size());

    const std::size_t full_key =
        make_hash_full(pars, lam_min, lam_max, lam_size, resOffset, resSlope);

    //std::cout << "[CompSynth] Made Hash. Spectrum Cache." << std::endl;     
    /* ===== 2nd-level cache (final spectrum) ======================== */
    SpectrumPtr final_sp =
        SpectrumCache::instance().insert_if_absent(full_key, [&] {

            /* ===== 1st-level cache (surface spectrum with rotation) === */
            const std::size_t surf_key =
                make_hash_surface(pars, resOffset, resSlope);

            SpectrumPtr surf_sp = SpectrumCache::instance()
                .insert_if_absent(surf_key, [&]{
                    // Now includes vsini for rotational broadening
                    return grid.load_spectrum(pars.teff, pars.logg,
                                               pars.z,   pars.he,
                                               pars.xi,  pars.vsini,
                                               resOffset, resSlope);
                });
            const Spectrum& surf = *surf_sp;      // safe reference
            //std::cout << "[CompSynth] Got Cached Spectrum. Doppler Shift." << std::endl;   
            /* ---------- remaining operations -------------------- */
            /* Rotational broadening is now already applied in grid.load_spectrum */
            
            /* 1) Doppler shift (depends on vrad) */
            constexpr double c = 299'792.458;           // km/s
            const double     factor = 1.0 + pars.vrad / c;
            Vector           lam_shift = surf.lambda * factor;
            
            //std::cout << "[CompSynth] Doppler Shifted. Interpolating onto wl grid." << std::endl;   

            /* 2) interpolate onto observed wavelength grid */
            Vector interp = interp_linear(lam_shift, surf.flux, lambda_obs);
            
            //std::cout << "[CompSynth] Interpolated. Finishing up." << std::endl;   
            /* 3) pack the final synthetic spectrum */
            Spectrum out;
            out.lambda = lambda_obs;
            out.flux   = std::move(interp);
            out.sigma  = Vector::Ones(lambda_obs.size());

            return out;    // moved into cache (as shared_ptr target)
        });

    /* Return a copy (keeps legacy API) */
    return *final_sp;
}

} // namespace specfit