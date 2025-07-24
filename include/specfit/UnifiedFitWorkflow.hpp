
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
    struct Config {
        int  n_outlier_iterations = 3;
        bool verbose              = true;
        bool debug_plots          = false;
        std::tuple<double, double> chi_thresholds = {-2.0, 2.0};

        
        /* NEW --------------------------------------------------------- */
        std::vector<std::string> untie_params;   // vrad,teff,…
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
    const std::vector<bool>& get_free_mask() const { return last_free_mask_; }

private:
    void solve_stage(const std::set<std::string>& free_params,
                     int                          max_iterations);

    void stage1_continuum_only();
    void stage2_continuum_vrad();
    void stage3_full();
    void stage4_outlier_rejection();
    void stage5_error_scaling();
    void stage6_final();

    void update_dataset_sigmas();

private:
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