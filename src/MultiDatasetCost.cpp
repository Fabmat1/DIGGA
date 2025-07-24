#include "specfit/MultiDatasetCost.hpp"
#include "specfit/ContinuumUtils.hpp"
#include <Eigen/Core>
#include <cmath>
#include <algorithm>

namespace specfit {

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
void MultiDatasetCost::compute_residuals(const Eigen::VectorXd& p,
                                         Eigen::VectorXd&       r) const
{
    r.setZero(num_residuals_);
    int row = 0;

    /* ---------- 1. loop over spectra -------------------------------- */
    for (std::size_t ds_idx = 0; ds_idx < datasets_.size(); ++ds_idx) {
        const auto& ds = datasets_[ds_idx];
        const int np = static_cast<int>(ds.lambda.size());
        const int na = ds.cont_param_count;

        /* a) continuum ------------------------------------------------ */
        Eigen::Map<const Vector> cont_y(p.data() +
                                        base_cont_offset_ +
                                        ds.cont_param_offset,
                                        na);
        Vector continuum = ds.cont_basis * cont_y;

        /* b) synthetic composite spectrum ---------------------------- */
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

            Spectrum s = compute_synthetic(*grids_[c],
                                            sp,
                                            ds.lambda,
                                            ds.resOffset,
                                            ds.resSlope);

            const double w = std::pow(sp.teff, 4);
            synth         += w * s.flux;
            w_sum         += w;
        }
        if (w_sum > 0.0) synth.array() /= w_sum;

        /* c) χ residuals  (skip ignored points) ---------------------- */
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
/*  residuals + Jacobian (numeric derivative for *all* parameters)       */
/* --------------------------------------------------------------------- */
void MultiDatasetCost::operator()(const Eigen::VectorXd& parameters,
                                  Eigen::VectorXd*       residuals,
                                  Eigen::MatrixXd*       jacobians) const
{
    if (residuals) {
        residuals->resize(num_residuals_);
        compute_residuals(parameters, *residuals);
    }
    if (!jacobians) return;

    const double eps_base = 1e-6;
    jacobians->resize(num_residuals_, n_total_params_);

    Eigen::VectorXd r0;
    compute_residuals(parameters, r0);

    for (int j = 0; j < n_total_params_; ++j) {
        /* multiplicative ε so it also works for small/large parameters */
        double h = eps_base * (std::abs(parameters[j]) + 1.0);
        Eigen::VectorXd p_eps = parameters;
        p_eps[j] += h;

        Eigen::VectorXd r_eps;
        compute_residuals(p_eps, r_eps);

        jacobians->col(j) = (r_eps - r0) / h;
    }
}

} // namespace specfit