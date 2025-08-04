#include "specfit/MultiDatasetCost.hpp"
#include "specfit/ContinuumUtils.hpp"
#include "specfit/Types.hpp"
#include <Eigen/Core>
#include <cmath>
#include <algorithm>

namespace specfit {

/* ------------------------------------------------------------------ */
/*  Cheap GEMV that is faster than OpenBLAS for “tall&skinny” basis   */
/* ------------------------------------------------------------------ */
static inline
Vector fast_continuum(const Eigen::MatrixXd& basis,
                      const Vector&          coeffs)      // (np × na) · (na)
{
    const int np = static_cast<int>(basis.rows());
    Vector     cont(np);

    cont.noalias()  = coeffs[0] * basis.col(0);
    for (int k = 1; k < coeffs.size(); ++k)
        cont.noalias() += coeffs[k] * basis.col(k);

    return cont;
}

MultiDatasetCost::MultiDatasetCost(const std::vector<DatasetInfo>& datasets,
                                   const std::vector<ModelGrid*>&  grids,
                                   int  n_components,
                                   const ParameterIndexer& indexer,
                                   int  /*total_residuals (unused)*/,
                                   int  total_cont_params)
    : datasets_(datasets)
    , grids_(grids)
    , n_components_(n_components)
    , n_total_params_(indexer.total_stellar_params + total_cont_params)
    , base_cont_offset_(indexer.total_stellar_params)
    , indexer_(indexer)
{
    /* how many points are *actually* fitted? -------------------------- */
    int kept_points = 0;
    for (const auto& ds : datasets_)
        for (size_t i = 0; i < ds.ignoreflag.size(); ++i)
            if (ds.ignoreflag[i] &&
                std::isfinite(ds.sigma[i]) &&
                ds.sigma[i] > 0.0)
                ++kept_points;

    num_residuals_ = kept_points;

    /* ----------  pre-compute Akima basis (one matrix per spectrum) ----- */
    for (auto& ds : datasets_) {
        const int np = static_cast<int>(ds.lambda.size());
        const int na = ds.cont_param_count;

        Eigen::MatrixXd basis(np, na);

        Vector unit = Vector::Zero(na);
        for (int k = 0; k < na; ++k) {
            unit.setZero();
            unit[k] = 1.0;
            AkimaSpline spl(ds.cont_x, unit);
            basis.col(k) = spl(ds.lambda);
        }
        ds.cont_basis = std::move(basis);
    }
}

/* --------------------------------------------------------------------- */
/*  residuals only (no derivatives)                                      */
/* --------------------------------------------------------------------- */
/* --------------------------------------------------------------------- */
/*  (1)  residuals only                                                  */
/* --------------------------------------------------------------------- */
void MultiDatasetCost::compute_residuals(const Eigen::VectorXd& p,
                                         Eigen::VectorXd&       r) const
{
    r.setZero(num_residuals_);
    int row = 0;

    /* ---------- loop over spectra ----------------------------------- */
    for (std::size_t ds_idx = 0; ds_idx < datasets_.size(); ++ds_idx) {
        const auto& ds  = datasets_[ds_idx];
        const int   np  = static_cast<int>(ds.lambda.size());
        const int   na  = ds.cont_param_count;

        /* ---- continuum -------------------------------------------- */
        Eigen::Map<const Vector> cont_y(p.data() +
                                        base_cont_offset_ +
                                        ds.cont_param_offset, na);
        const Vector continuum = fast_continuum(ds.cont_basis, cont_y);

        /* ---- synthetic composite spectrum ------------------------- */
        Vector synth  = Vector::Zero(np);
        double w_sum  = 0.0;

        for (int c = 0; c < n_components_; ++c) {
            StellarParams sp;
            const int d = static_cast<int>(ds_idx);
            sp.vrad  = p[indexer_.get(c,d,0)];
            sp.vsini = p[indexer_.get(c,d,1)];
            sp.zeta  = p[indexer_.get(c,d,2)];
            sp.teff  = p[indexer_.get(c,d,3)];
            sp.logg  = p[indexer_.get(c,d,4)];
            sp.xi    = p[indexer_.get(c,d,5)];
            sp.z     = p[indexer_.get(c,d,6)];
            sp.he    = p[indexer_.get(c,d,7)];

            Spectrum s = compute_synthetic(*grids_[c], sp,
                                            ds.lambda,
                                            ds.resOffset, ds.resSlope);

            const double w = std::pow(sp.teff, 4);
            synth         += w * s.flux;
            w_sum         += w;
        }
        if (w_sum > 0.0) synth.array() /= w_sum;

        /* ---- χ residuals  (skip ignored points) ------------------- */
        for (int i = 0; i < np; ++i) {
            if (!ds.ignoreflag[i]) continue;

            const double sigma = ds.sigma[i];
            if (!std::isfinite(sigma) || sigma <= 0.0) continue;

            const double model = synth[i] * continuum[i];
            r[row++]           = (model - ds.flux[i]) / sigma;
        }
    }
}

/* --------------------------------------------------------------------- */
/*  (2)  residuals + Jacobian                                            */
/* --------------------------------------------------------------------- */
void MultiDatasetCost::operator()(const Eigen::VectorXd& parameters,
                                  Eigen::VectorXd*       residuals,
                                  Eigen::MatrixXd*       jacobians) const
{
    /* ========== residual vector ==================================== */
    if (residuals) {
        residuals->resize(num_residuals_);
        compute_residuals(parameters, *residuals);
    }

    if (!jacobians) return;                       // user wants resid only
    jacobians->setZero(num_residuals_, n_total_params_);

    /* ========== common data for analytic continuum columns ========= */
    std::vector<Vector>  all_synth;  // one per dataset (to reuse later)
    std::vector<Vector>  all_sigma;

    all_synth.reserve(datasets_.size());
    all_sigma.reserve(datasets_.size());

    for (const auto& ds : datasets_)
        all_sigma.push_back(ds.sigma);            // cheap copy of ref

    /* ---------------------------------------------------------------
       Build synthetic spectra & keep them – we need them twice:
       (a) for analytic Jacobian columns
       (b) for residual FD later (saves recomputation)
    ---------------------------------------------------------------- */
    for (std::size_t ds_idx = 0; ds_idx < datasets_.size(); ++ds_idx) {
        const auto& ds  = datasets_[ds_idx];
        const int   np  = static_cast<int>(ds.lambda.size());

        Vector synth  = Vector::Zero(np);
        double w_sum  = 0.0;

        for (int c = 0; c < n_components_; ++c) {
            StellarParams sp;
            const int d = static_cast<int>(ds_idx);
            sp.vrad  = parameters[indexer_.get(c,d,0)];
            sp.vsini = parameters[indexer_.get(c,d,1)];
            sp.zeta  = parameters[indexer_.get(c,d,2)];
            sp.teff  = parameters[indexer_.get(c,d,3)];
            sp.logg  = parameters[indexer_.get(c,d,4)];
            sp.xi    = parameters[indexer_.get(c,d,5)];
            sp.z     = parameters[indexer_.get(c,d,6)];
            sp.he    = parameters[indexer_.get(c,d,7)];

            Spectrum s = compute_synthetic(*grids_[c], sp,
                                            ds.lambda,
                                            ds.resOffset, ds.resSlope);

            const double w = std::pow(sp.teff, 4);
            synth         += w * s.flux;
            w_sum         += w;
        }
        if (w_sum > 0.0) synth.array() /= w_sum;
        all_synth.emplace_back(std::move(synth));
    }

    /* ========== (A) analytic continuum columns ===================== */
    int row_global = 0;
    for (std::size_t ds_idx = 0; ds_idx < datasets_.size(); ++ds_idx) {
        const auto& ds  = datasets_[ds_idx];
        const int   np  = static_cast<int>(ds.lambda.size());
        const int   na  = ds.cont_param_count;

        /* -- current continuum vector ------------------------------- */
        Eigen::Map<const Vector> cont_y(parameters.data() +
                                        base_cont_offset_ +
                                        ds.cont_param_offset, na);
        const Vector continuum = fast_continuum(ds.cont_basis, cont_y);

        /* -- per-anchor loop ---------------------------------------- */
        for (int k = 0; k < na; ++k) {
            const int j_global = base_cont_offset_ + ds.cont_param_offset + k;
            int row = row_global;

            for (int i = 0; i < np; ++i) {
                if (!ds.ignoreflag[i]) continue;
                const double sigma = all_sigma[ds_idx][i];
                if (!std::isfinite(sigma) || sigma <= 0.0) continue;

                /* d(model)/da_k  =  synth * basis_ik                    */
                const double dmodel = all_synth[ds_idx][i] * ds.cont_basis(i,k);
                jacobians->coeffRef(row, j_global) = dmodel / sigma;
                ++row;
            }
        }

        /* advance residual row offset ------------------------------- */
        for (int i = 0; i < np; ++i)
            if (ds.ignoreflag[i] &&
                std::isfinite(ds.sigma[i]) &&
                ds.sigma[i] > 0.0)
                ++row_global;
    }

    /* ========== (B) FD for stellar parameters only ================= */
    const double eps_base = 1e-6;
    Eigen::VectorXd r0;
    compute_residuals(parameters, r0);

    for (int j = 0; j < base_cont_offset_; ++j)   // stellar parameters
    {
        double h = eps_base * (std::abs(parameters[j]) + 1.0);
        Eigen::VectorXd p_eps = parameters;
        p_eps[j] += h;

        Eigen::VectorXd r_eps;
        compute_residuals(p_eps, r_eps);

        jacobians->col(j) = (r_eps - r0) / h;
    }
}

} // namespace specfit