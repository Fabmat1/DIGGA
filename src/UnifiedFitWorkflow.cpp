#include "specfit/UnifiedFitWorkflow.hpp"
#include "specfit/MultiDatasetCost.hpp"
#include "specfit/SimpleLM.hpp"
#include "specfit/ReportUtils.hpp"
#include <filesystem>
#include <iostream>
#include <set>
#include <numeric>

namespace specfit {

/* ------------------------------------------------------------------------- */
/*  constructor – unchanged (only removed any Ceres include)                 */
/* ------------------------------------------------------------------------- */
UnifiedFitWorkflow::UnifiedFitWorkflow(
        std::vector<DataSet>& datasets,
        SharedModel&          model,
        const Config&         config,
        const std::vector<std::map<std::string,bool>>& frozen_status,
        int                   nthreads)
    : datasets_(datasets)
    , model_(model)
    , config_(config)
    , frozen_status_(frozen_status)
    , nthreads_(nthreads)
{
    /* ================================================================ */
    /*  0.  build stellar-parameter indexer                            */
    /* ================================================================ */
    const int n_components = static_cast<int>(model_.params.size());
    const int n_datasets   = static_cast<int>(datasets_.size());
    
    indexer_.build(n_components, n_datasets, config_.untie_params);
    
    /* ----------  collect initial parameters into one big vector -------- */
    unified_params_.resize(indexer_.total_stellar_params);
    for (int c = 0; c < n_components; ++c) {
        const auto& sp = model_.params[c];
        for (int d = 0; d < n_datasets; ++d) {
            unified_params_[ indexer_.get(c,d,0) ] = sp.vrad ;
            unified_params_[ indexer_.get(c,d,1) ] = sp.vsini;
            unified_params_[ indexer_.get(c,d,2) ] = sp.zeta ;
            unified_params_[ indexer_.get(c,d,3) ] = sp.teff ;
            unified_params_[ indexer_.get(c,d,4) ] = sp.logg ;
            unified_params_[ indexer_.get(c,d,5) ] = sp.xi   ;
            unified_params_[ indexer_.get(c,d,6) ] = sp.z    ;
            unified_params_[ indexer_.get(c,d,7) ] = sp.he   ;
        }
    }
    for (const auto& ds : datasets_)
        unified_params_.insert(unified_params_.end(),
                               ds.cont_y.begin(), ds.cont_y.end());
}

/* ------------------------------------------------------------------------- */
/*  helper that performs one optimisation stage                              */
/* ------------------------------------------------------------------------- */
void UnifiedFitWorkflow::solve_stage(const std::set<std::string>& free_params,
                                     int max_iterations)
{
    /* ---- a) gather bookkeeping info ----------------------------------- */
    static int dbg_stage_counter = 0;
    const int n_components  = static_cast<int>(model_.params.size());
    const int stellar_total = indexer_.total_stellar_params;

    std::vector<DatasetInfo> ds_infos;
    std::vector<ModelGrid*>  grid_ptrs;
    int total_residuals = 0;
    int cont_offset     = 0;

    for (auto& ds : datasets_) {
        DatasetInfo info;
        info.lambda            = ds.obs.lambda;
        info.flux              = ds.obs.flux;
        info.sigma             = ds.obs.sigma;
        info.ignoreflag        = ds.obs.ignoreflag;
        info.cont_x            = ds.cont_x;
        info.resOffset         = ds.resOffset;
        info.resSlope          = ds.resSlope;
        info.cont_param_offset = cont_offset;
        info.cont_param_count  = ds.cont_y.size();

        ds_infos.push_back(std::move(info));
        const int n_kept = std::accumulate(ds.obs.ignoreflag.begin(),
                               ds.obs.ignoreflag.end(), 0);
        total_residuals += n_kept;
        cont_offset     += ds.cont_y.size();
    }
    for (auto& g : model_.grids) grid_ptrs.push_back(&g);

    MultiDatasetCost cost(ds_infos, grid_ptrs, n_components,
                              indexer_,
                               total_residuals, cont_offset);

    const int Npar = stellar_total + cont_offset;

    /* ---- b)   build lower / upper bounds ------------------------------ */
    std::vector<double> lo(Npar, -1.0e10);
    std::vector<double> hi(Npar,  1.0e10);

    const char* names[8] = { "vrad","vsini","zeta","teff",
                             "logg","xi","z","he" };
    
    for (int c = 0; c < n_components; ++c)
        for (std::size_t d = 0; d < datasets_.size(); ++d)
            for (int p = 0; p < 8; ++p)
            {
                int idx = indexer_.get(c,static_cast<int>(d),p);
                switch(p){
                    case 0: lo[idx] = -1000.0; hi[idx] = 1000.0; break; // vrad
                    case 1: lo[idx] =     0.0; hi[idx] =  500.0; break; // vsini
                    case 3: lo[idx] =  3000.0; hi[idx] = 50000.0;break; // teff
                    case 4: lo[idx] =     0.0; hi[idx] =     9.0;break; // logg
                    default: break; // others unlimited
                }
            }

    /* ---- c)   decide which parameters are free ------------------------ */
    std::vector<bool> free_mask(Npar, false);

    auto mark_free = [&](int global_idx) { free_mask[global_idx] = true; };
    
    const bool all_requested = free_params.count("all") > 0;

    // stellar params component-wise
    for (int c = 0; c < n_components; ++c)
    {
        auto& frz = frozen_status_[c];

        auto token = [&](const std::string& name){ return "c"+std::to_string(c+1)+"_"+name; };
        for (std::size_t d = 0; d < datasets_.size(); ++d)
            for (int p = 0; p < 8; ++p)
            {
                if (!(all_requested ||
                      free_params.count(token(names[p])))) continue;
                if (frz.at(names[p])) continue;
                mark_free(indexer_.get(c,static_cast<int>(d),p));
            }
    }

    // continuum anchors
    if (free_params.count("continuum"))
        for (int i = stellar_total; i < Npar; ++i)
            mark_free(i);

    /* ---- d)   run Levenberg–Marquardt -------------------------------- */
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(unified_params_.data(), Npar);

    LMSolverOptions opt;
    opt.max_iterations = max_iterations;
    opt.verbose        = config_.verbose;

    summary_ = levenberg_marquardt(
            [&cost](const Eigen::VectorXd& p,
                    Eigen::VectorXd*       r,
                    Eigen::MatrixXd*       J)
            { cost(p, r, J); },
            x, free_mask, lo, hi, opt);
    last_free_mask_ = free_mask;                            

    Eigen::Map<Eigen::VectorXd>(unified_params_.data(), Npar) = x;
    /* -------------------------  debug plots  ------------------------- */
    if (config_.debug_plots) {
        namespace fs = std::filesystem;
        fs::create_directories("debug");

        MultiPanelPlotter P(1.0, false);          // xrange ignored by simple_plot()
        for (std::size_t d = 0; d < datasets_.size(); ++d) {
            Vector mdl = get_model_for_dataset(d);
            std::string pdf =
                "debug/stage" + std::to_string(dbg_stage_counter) + "_" +
                fs::path(datasets_[d].name).stem().string() + ".pdf";
            P.simple_plot(pdf,
                          datasets_[d].obs,
                          mdl);
        }
    }
    ++dbg_stage_counter;   
}

/* ------------------------------------------------------------------------- */
/*  small wrappers for the six stages                                        */
/* ------------------------------------------------------------------------- */
void UnifiedFitWorkflow::stage1_continuum_only() {
    solve_stage( { "continuum" }, 100 );
}

void UnifiedFitWorkflow::stage2_continuum_vrad() {
    std::set<std::string> fp = { "continuum" };
    for (std::size_t c = 0; c < model_.params.size(); ++c)
        fp.insert("c"+std::to_string(c+1)+"_vrad");

    solve_stage(fp, 100);
}

void UnifiedFitWorkflow::stage3_full() {
    std::set<std::string> fp = { "all", "continuum" };
    solve_stage(fp, 200);
}


/* ------------------------------------------------------------------------- */
/*  Stage-4 : iterative 10-σ outlier rejection + re-scaling                  */
/* ------------------------------------------------------------------------- */
void UnifiedFitWorkflow::stage4_outlier_rejection()
{
    bool any_flagged = true;
    int  iteration   = 0;

    while (any_flagged) {
        any_flagged = false;
        ++iteration;

        /* ---------- inspect residuals, flag > 10-σ points ---------------- */
        for (std::size_t d = 0; d < datasets_.size(); ++d) {
            Vector model = get_model_for_dataset(d);
            auto&  ds    = datasets_[d];

            for (int i = 0; i < ds.obs.flux.size(); ++i) {
                if (!ds.obs.ignoreflag[i]) continue;        // already rejected
                double res = (model[i] - ds.obs.flux[i]) / ds.obs.sigma[i];
                if (std::abs(res) > 10.0) {
                    ds.obs.ignoreflag[i] = 0;               // throw out
                    any_flagged = true;
                }
            }
        }
    }

    if (config_.verbose)
        std::cout << "[Stage 5]  finished after " << iteration
                  << " iteration(s)\n";
}

/* ------------------------------------------------------------------------- */
/*  Stage-5 : global error rescaling (χ² / dof → 1)                          */
/* ------------------------------------------------------------------------- */
void UnifiedFitWorkflow::stage5_error_scaling()
{
    using std::fabs;
    using std::sqrt;
    using std::isfinite;

    /* ------------------------------------------------------------------
       0.  Remember *original* σ only once (deep copies).
           -------------------------------------------------------------- */
    static std::vector<Vector> original_sigmas;
    if (original_sigmas.empty()) {
        original_sigmas.reserve(datasets_.size());
        for (const auto &ds : datasets_)
            original_sigmas.push_back(ds.obs.sigma);
    }

    /* ------------------------------------------------------------------
       1.  Per-pixel outlier handling  (σ_i → σ_i · |χ_i| / χ_thres)
           -------------------------------------------------------------- */
    const double chi_thres_pos = (std::get<1>(config_.chi_thresholds) != 0) ? fabs(std::get<1>(config_.chi_thresholds)) : 2.0;
    const double chi_thres_neg = (std::get<0>(config_.chi_thresholds) != 0) ? -fabs(std::get<0>(config_.chi_thresholds)) : -2.0;

    //  keep a second work-copy of σ that we will update step by step
    std::vector<Vector> work_sigma = original_sigmas;

    std::size_t n_kept = 0;            // number of pixels entering χ²
    double      chi2   = 0.0;          // accumulated χ² after step (1)

    for (std::size_t d = 0; d < datasets_.size(); ++d)
    {
        const Vector model = get_model_for_dataset(d);
        const auto  &flux  = datasets_[d].obs.flux;

        for (int i = 0; i < flux.size(); ++i)
        {
            if (datasets_[d].obs.ignoreflag[i] == 0)                          continue;
            const double sig0 = original_sigmas[d][i];
            if (!isfinite(sig0) || sig0 <= 0.0)                               continue;

            /* ----- χ_i with the ORIGINAL σ_i --------------------------- */
            const double chi_i = (flux[i] - model[i]) / sig0;

            double fac = 1.0;
            if (chi_i >  chi_thres_pos) fac = fabs(chi_i) /  chi_thres_pos;
            if (chi_i <  chi_thres_neg) fac = fabs(chi_i) / -chi_thres_neg;

            work_sigma[d][i] = sig0 * fac;

            /* ----- accumulate χ² with the *enlarged* σ ---------------- */
            const double diff = flux[i] - model[i];
            chi2 += (diff * diff) / (work_sigma[d][i] * work_sigma[d][i]);
            ++n_kept;
        }
    }
    if (n_kept == 0) return;                       // nothing noticed

    /* ------------------------------------------------------------------
       2.  Global rescaling so that χ²_red = 1 
           -------------------------------------------------------------- */
    
    // Count the number of free parameters from the last optimization stage
    const int n_free_params = std::count(last_free_mask_.begin(), last_free_mask_.end(), true);
    
    const double dof = static_cast<double>(n_kept) - static_cast<double>(n_free_params);
    if (dof > 0.0) {
        const double factor     = sqrt(chi2 / dof);      // = √χ²_red
        if (factor > 0.0 && fabs(factor - 1.0) > 1e-12)  // avoid work
            for (auto &vec : work_sigma)
                for (double &s : vec) s *= factor;
    }

    /* ------------------------------------------------------------------
       3.  Commit the new σ to the data sets
           -------------------------------------------------------------- */
    for (std::size_t d = 0; d < datasets_.size(); ++d)
        datasets_[d].obs.sigma = work_sigma[d];

}

void UnifiedFitWorkflow::stage6_final() { stage3_full(); }

/* ------------------------------------------------------------------------- */
/*  public “run” orchestrator                                                */
/* ------------------------------------------------------------------------- */
void UnifiedFitWorkflow::run()
{
    std::cout << "[Stage 1] continuum only …\n";   stage1_continuum_only();
    std::cout << "[Stage 2] continuum + vrad …\n"; stage2_continuum_vrad();
    std::cout << "[Stage 3] full fit …\n";        stage3_full();
    std::cout << "[Stage 4] outlier rejection …\n";stage4_outlier_rejection();
    std::cout << "[Stage 5] error scaling …\n";   stage5_error_scaling();
    std::cout << "[Stage 6] final fit …\n";       stage6_final();
    final_uncertainties_ = summary_.param_uncertainties;   // 

    /* update model structure with the final parameter values */
    for (std::size_t c = 0; c < model_.params.size(); ++c) {
        /* for tied parameters the value is identical for all spectra
           – take the first dataset as representative               */
        model_.params[c].vrad  = unified_params_[ indexer_.get(c,0,0) ];
        model_.params[c].vsini = unified_params_[ indexer_.get(c,0,1) ];
        model_.params[c].zeta  = unified_params_[ indexer_.get(c,0,2) ];
        model_.params[c].teff  = unified_params_[ indexer_.get(c,0,3) ];
        model_.params[c].logg  = unified_params_[ indexer_.get(c,0,4) ];
        model_.params[c].xi    = unified_params_[ indexer_.get(c,0,5) ];
        model_.params[c].z     = unified_params_[ indexer_.get(c,0,6) ];
        model_.params[c].he    = unified_params_[ indexer_.get(c,0,7) ];
    }
}


Vector UnifiedFitWorkflow::get_model_for_dataset(size_t dataset_idx) const {
    if (dataset_idx >= datasets_.size()) {
        throw std::out_of_range("Invalid dataset index");
    }
    
    const auto& ds = datasets_[dataset_idx];
    const int n_points = ds.obs.lambda.size();
    
    // Extract current stellar parameters (dataset-specific)
    std::vector<StellarParams> stellar(model_.params.size());
    const int didx = static_cast<int>(dataset_idx);
    for (int c = 0; c < static_cast<int>(model_.params.size()); ++c) {
        stellar[c].vrad  = unified_params_[ indexer_.get(c,didx,0) ];
        stellar[c].vsini = unified_params_[ indexer_.get(c,didx,1) ];
        stellar[c].zeta  = unified_params_[ indexer_.get(c,didx,2) ];
        stellar[c].teff  = unified_params_[ indexer_.get(c,didx,3) ];
        stellar[c].logg  = unified_params_[ indexer_.get(c,didx,4) ];
        stellar[c].xi    = unified_params_[ indexer_.get(c,didx,5) ];
        stellar[c].z     = unified_params_[ indexer_.get(c,didx,6) ];
        stellar[c].he    = unified_params_[ indexer_.get(c,didx,7) ];
    }
    
    /* -------- continuum anchors live in ONE big block at the very end --- */
    int total_cont = 0;
    for (const auto& dsi : datasets_) total_cont += dsi.cont_y.size();
    
    const int cont_block_start = static_cast<int>(unified_params_.size()) - total_cont;
    
    int cont_offset = 0;
    for (std::size_t j = 0; j < dataset_idx; ++j)
        cont_offset += datasets_[j].cont_y.size();
    
    Vector cont_y = Eigen::Map<const Vector>(
        unified_params_.data() + cont_block_start + cont_offset,
        ds.cont_y.size());
    
    // Build continuum
    AkimaSpline cont_spline(ds.cont_x, cont_y);
    Vector continuum = cont_spline(ds.obs.lambda);
    
    // Compute synthetic spectrum
    Vector model = Vector::Zero(n_points);
    double weight_sum = 0.0;
    
    for (size_t c = 0; c < model_.grids.size(); ++c) {
        Spectrum synth = compute_synthetic(
            model_.grids[c], stellar[c], ds.obs.lambda,
            ds.resOffset, ds.resSlope);
        
        double weight = std::pow(stellar[c].teff, 4);
        model += weight * synth.flux;
        weight_sum += weight;
    }
    
    if (weight_sum > 0) {
        model /= weight_sum;
    }
    
    // Apply continuum
    return model.cwiseProduct(continuum);
}

void UnifiedFitWorkflow::update_dataset_sigmas() {
    for (size_t d = 0; d < datasets_.size(); ++d) {
        Vector model      = get_model_for_dataset(d);
        Vector residuals  = (model - datasets_[d].obs.flux)
                            .cwiseQuotient(datasets_[d].obs.sigma);

        for (int i = 0; i < residuals.size(); ++i) {
            if (std::abs(residuals[i]) > 10.0 && datasets_[d].obs.ignoreflag[i]) {
                datasets_[d].obs.ignoreflag[i] = 0;         // flag as “bad”
                if (i < datasets_[d].keep.size())           // keep old plotting support
                    datasets_[d].keep[i] = 0;
            }
        }
    }
}

} // namespace specfit