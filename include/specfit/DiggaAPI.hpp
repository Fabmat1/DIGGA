#pragma once
#include "Types.hpp"
#include <array>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace specfit { class UnifiedFitWorkflow; }   // forward decl in parent ns

namespace specfit::api {

// ---------- Global settings (one per DiggaSession) -----------------------
struct GlobalSettings {
    std::vector<std::string> base_paths;   // where to look up model grids

    // reporting / gui-only
    double xrange         = 500.0;

    // spectrum rejection (applied during preprocessing)
    double filter_snr     = 0.0;   // reject if SNR <= filter_snr (0 disables)
    double require_blue   = 0.0;   // reject if lambda_min >= require_blue (0 disables)

    // stage behaviour
    bool   auto_freeze_vsini = true;
    std::vector<std::string> untie_params = {"vrad"};

    // iterative-noise / outlier rejection (stage 6)
    int    nit_noise_max  = 5;
    int    nit_fit_max    = 5;
    int    width_box_px   = 5;
    double outlier_sigma_lo = 3.0;
    double outlier_sigma_hi = 3.0;
    double conv_range_lo  = 0.9;
    double conv_range_hi  = 1.1;
    double conv_fraction  = 0.9;

    bool   verbose        = true;
    bool   debug_plots    = false;

    class UnifiedFitWorkflow;   // forward decl
    std::function<void(int stage_index,
        const ::specfit::UnifiedFitWorkflow& wf)>
    on_stage_complete;
};

// ---------- Per-fit input -------------------------------------------------
struct StellarComponentInit {
    std::string grid_relative_path;        // e.g. "sdB/processed/"
    double vrad=0, vsini=0, zeta=0;
    double teff=0, logg=0, xi=0, z=0, he=0;
    bool freeze_vrad=false, freeze_vsini=false, freeze_zeta=true;
    bool freeze_teff=false, freeze_logg=false, freeze_xi=true;
    bool freeze_z=false,    freeze_he=false;
};

struct SpectrumFileInput {
    std::string filename;
    std::string spectype;                  // "ASCII_with_2_columns", ...
    double resOffset = 0.0;
    double resSlope  = 0.0;
    double barycorr  = 0.0;

    // optional per-file overrides (otherwise inherit from ObservationInput)
    std::optional<std::array<double,2>>                  waveCut;
    std::optional<std::vector<std::array<double,2>>>     ignore;
    std::optional<std::vector<std::array<double,3>>>     cspline_anchorpoints;
};

struct ObservationInput {
    std::vector<SpectrumFileInput> files;
    std::array<double,2> waveCut{ -1e300, 1e300 };
    std::vector<std::array<double,2>> ignore;
    std::vector<std::array<double,3>> cspline_anchorpoints;
};

struct FitInput {
    std::vector<StellarComponentInit> components;   // c1, c2, ...
    std::vector<ObservationInput>     observations;
    std::string                       output_path;  // optional; used if reports requested
};

// ---------- Results -------------------------------------------------------
struct StellarParamResult {
    double value  = 0.0;
    double error  = 0.0;   // 0 if frozen
    bool   frozen = false;
    bool   at_boundary = false;
};

struct ComponentResult {
    // each of these may be 1 (tied) or n_spectra (untied) long
    std::vector<StellarParamResult> vrad, vsini, zeta, teff, logg, xi, z, he;
};

struct SpectrumResult {
    std::string source_filename;
    // rebinned observed spectrum on the Nyquist grid (what the fit saw)
    Vector lambda;
    Vector flux;
    Vector sigma;
    std::vector<int> ignoreflag;   // 1 = used, 0 = ignored

    // synthetic model (continuum * stellar) on the same lambda grid
    Vector model;
    // fitted continuum spline evaluated on the same lambda grid
    Vector continuum;

    // continuum-spline anchors
    Vector cont_x;
    Vector cont_y;
};

struct FitResult {
    bool   converged   = false;
    int    iterations  = 0;
    double final_chi2  = 0.0;
    int    n_free_parameters = 0;
    int    n_data_points     = 0;

    std::vector<ComponentResult> components;
    std::vector<SpectrumResult>  spectra;

    // Files rejected during preprocessing (SNR / blue-cut / load failure)
    std::vector<std::string> rejected_files;

    // raw flat parameter vector + uncertainties (for advanced consumers)
    std::vector<double> raw_params;
    std::vector<double> raw_uncertainties;
    std::vector<bool>   raw_free_mask;
};

// ---------- The session object --------------------------------------------
class DiggaSession {
public:
    DiggaSession();
    ~DiggaSession();

    void set_global_settings(const GlobalSettings& gs);
    void set_fit_input     (const FitInput& fi);

    void set_num_threads(int n);      // 0 = hardware_concurrency

    // Optional progress callback: receives stage name and fractional progress [0,1]
    using ProgressFn = std::function<void(const std::string& stage, double frac)>;
    void set_progress_callback(ProgressFn cb);

    // Optional log-line callback: receives a single log line (no trailing newline)
    using LogFn = std::function<void(const std::string& line)>;
    void set_log_callback(LogFn cb);

    // Runs preprocessing + UnifiedFitWorkflow + collects results.
    // Throws std::runtime_error on fatal errors.
    FitResult run();

    // If you want to emit the LaTeX/PDF report + plots, call after run().
    // Requires DIGGAreport to be linked. Throws if that target wasn't built.
    void write_report(const FitResult& r, const std::string& out_dir,
                      bool make_plots = true, bool make_pdf = true) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------- JSON adapters (optional; in separate TU) ----------------------
//   These let ASTRA reuse your existing .json files without duplicating code.
GlobalSettings global_settings_from_json_file(const std::string& path);
FitInput       fit_input_from_json_file      (const std::string& path);

} // namespace specfit::api
