#pragma once
#include "Types.hpp"
#include "Spectrum.hpp"
#include "ModelGrid.hpp"
#include "SyntheticModel.hpp"
#include <vector>
#include <string>

namespace specfit {

// One observed spectrum dataset
struct DataSet {
    std::string              name;        // e.g. file name (for diagnostics)
    Spectrum                 obs;         // on Nyquist grid, *after* masking
    Vector                   cont_x;      // anchor abscissae
    std::vector<double>      cont_y;      // mutable ordinates (parameters)
    std::vector<int>         keep;        // 1 = use, 0 = ignored  (same size as obs.lambda)
    double                   resOffset = 0.0;
    double                   resSlope  = 0.0;
    
    // Helper to get current continuum parameter offset in unified parameter vector
    int cont_param_offset = 0;
};

// Bundle of model grids and parameters shared across all datasets
struct SharedModel {
    std::vector<ModelGrid>      grids;
    std::vector<StellarParams>  params;
};

// User options that control the staged fitting
struct FitPlan {
    bool freeze_vrad_initial  = true;   // stage-1 : cont only
    bool freeze_vsini_initial = true;
    bool vrad_free_in_stage2  = true;   // stage-2 : cont + optional vrad
    int  n_outlier_iter       = 3;
    bool verbose              = true;   // Ceres prints every iteration
    bool debug_plots          = true;   // write PNG per stage
};

} // namespace specfit