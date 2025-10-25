
#pragma once
#include "Types.hpp"
#include "CommonTypes.hpp"
#include "ModelGrid.hpp"
#include "SimpleLM.hpp"
#include <vector>
#include <memory>
#include <set>
#include <map>
#include "ParameterIndexer.hpp"
#include <tuple>

namespace specfit {

class UnifiedFitWorkflow {
public:
    struct Config
    {
        // ---------------- previous fields ----------------
        int    n_outlier_iterations = 3;            // kept (stage-1 stuff)
        bool   verbose              = true;
        bool   debug_plots          = false;
        std::tuple<double,double> chi_thresholds = {-2.0, 2.0};
        std::vector<std::string> untie_params;      // vrad, …

        // ---------------- NEW : S-Lang iterative noise -----------------
        int    nit_noise_max            = 5;      // outer passes
        int    nit_fit_max              = 5;      // micro-fits per pass
        int    width_box_px             = 5;      // neighbourhood for σ
        double outlier_sigma_lo         = 2.0;    // −2 σ
        double outlier_sigma_hi         = 2.0;    // +2 σ
        double conv_range_lo            = 0.9;    // 0.9 < s < 1.1
        double conv_range_hi            = 1.1;
        double conv_fraction            = 0.9;    // 90 % of bins
    };

    UnifiedFitWorkflow(std::vector<DataSet>& datasets,
                       SharedModel&           model,
                       const Config&          config,
                       const std::vector<std::map<std::string,bool>>& frozen_status,
                       int                    nthreads);

    void run();

    Vector get_model_for_dataset(std::size_t dataset_idx) const;
    const LMSolverSummary& get_summary() const { return summary_; }
    const std::vector<double>& get_parameters()   const { return unified_params_; }
    const std::vector<double>& get_uncertainties() const { return final_uncertainties_; }
    const double& get_final_chi2() const { return summary_.final_chi2; }
    const std::vector<bool>& get_free_mask() const { return last_free_mask_; }

private:
    void solve_stage(const std::set<std::string>& free_params,
                     int                          max_iterations,
                     bool                         add_powell = false);

    void stage1_continuum_only();
    void stage2_continuum_vrad();
    void stage3_continuum_vrad_teff_logg_z();
    void stage4_full(bool add_powell = false);
    void stage5_auto_freeze_vsini();
    void stage6_rescale_and_reject();
    void stage7_final();
    
    void report_boundary_parameters() const;
    double chi2_current() const;      //  <──  new

private:

    LMWorkspace lm_mem_;   // lives as long as the workflow lives
    std::vector<DataSet>& datasets_;
    SharedModel&          model_;
    Config                config_;
    std::vector<std::map<std::string,bool>> frozen_status_;
    int                   nthreads_;

    /* --- NEW : stellar-parameter mapping ---------------------------- */
    ParameterIndexer      indexer_;

    std::vector<double>   unified_params_;
    LMSolverSummary       summary_;
    std::vector<double>   final_uncertainties_;   // filled after stage 6
    std::vector<bool>  last_free_mask_;  
};


} // namespace specfit