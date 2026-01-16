#include "specfit/JsonUtils.hpp"
#include "specfit/UnifiedFitWorkflow.hpp"
#include "specfit/ContinuumUtils.hpp"
#include "specfit/NyquistGrid.hpp"
#include "specfit/Rebin.hpp"
#include "specfit/ReportUtils.hpp"
#include "specfit/CommonTypes.hpp"
#include "specfit/SpectrumLoaders.hpp"
#include "specfit/SpectrumCache.hpp"
#include "specfit/SyntheticModel.hpp"
#include <cxxopts.hpp>
#include <Eigen/Core>
#include <iostream>
#include <omp.h>
#include <algorithm>
#include <cassert>
#include <filesystem>
#include "matplotlibcpp.h"
#include <chrono>
#include <thread>
#include <fstream>

namespace fs = std::filesystem;
using namespace specfit;

// Function to find and load the JSON config
nlohmann::json load_global_config() {
    std::vector<std::string> search_paths = {
        // 1. Current working directory
        "global_settings.json",
        
        // 2. Same directory as executable
        []() {
            try {
                auto exe_path = std::filesystem::canonical("/proc/self/exe");
                return (exe_path.parent_path() / "global_settings.json").string();
            } catch (...) {
                return std::string("./global_settings.json");
            }
        }(),
        
        // 3. Relative to executable (alternative method)
        []() {
            try {
                auto exe_path = std::filesystem::read_symlink("/proc/self/exe");
                return (exe_path.parent_path() / "global_settings.json").string();
            } catch (...) {
                return std::string("./global_settings.json");
            }
        }(),
        
        // 4. Build directory (for development)
        "../global_settings.json",
        
        // 5. Source directory (fallback)
        "../../global_settings.json"
    };
    
    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path)) {
            std::ifstream file(path);
            if (file.is_open()) {
                try {
                    nlohmann::json config;
                    file >> config;
                    std::cout << "Loaded config from: " << path << std::endl;
                    return config;
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing JSON from " << path << ": " << e.what() << std::endl;
                    continue;
                }
            }
        }
    }
    
    throw std::runtime_error("Could not find or load global_settings.json");
}

int main(int argc, char** argv) {
    auto start_time = std::chrono::steady_clock::now();
    try {
        cxxopts::Options opts("specfit", "Multi-dataset stellar spectrum fitting");
        opts.add_options()
            ("fit", "Fit configuration JSON", cxxopts::value<std::string>())
            ("threads", "Number of threads", cxxopts::value<int>()->default_value("0"))
            ("output-synthetic", "Output normalized synthetic spectrum without degradation")
            ("cache-size", "Maximum number of cache entries", cxxopts::value<int>()->default_value("100"))
            ("debug-plots", "Enable debug plotting output")
            ("no-plots",   "Skip creation of per-spectrum plots")     //  <-- NEW
            ("no-pdf",     "Do not run pdflatex on the result .tex")   //  <-- NEW
            ("h,help", "Show help");
        
        auto cli = opts.parse(argc, argv);
        if (cli.count("help") || (!cli.count("fit") && !cli.count("output-synthetic"))) {
            std::cout << opts.help() << '\n';
            return 0;
        }

        specfit::SpectrumCache::instance().set_capacity(cli["cache-size"].as<int>());
        
        // Load configurations
        auto global_cfg = load_global_config();
        auto fit_cfg = load_json(cli["fit"].as<std::string>());
        expand_env(global_cfg);
        expand_env(fit_cfg);
        
        // Setup
        
        int nthreads = cli["threads"].as<int>();
        if (nthreads <= 0) nthreads = std::thread::hardware_concurrency();

        omp_set_num_threads(nthreads);
        Eigen::setNbThreads(nthreads);

        std::vector<std::string> base_paths = 
            global_cfg["basePaths"].get<std::vector<std::string>>();
        
        // Load model grids
        auto grid_names = fit_cfg["grids"].get<std::vector<std::string>>();
        SharedModel model;
        for (const auto& g : grid_names) {
            model.grids.emplace_back(base_paths, g);
        }
        
        // Initialize parameters
        model.params.resize(grid_names.size());
        std::vector<std::map<std::string, bool>> frozen_status(grid_names.size());
        
        auto init_guess = fit_cfg["initialGuess"];
        for (size_t c = 0; c < model.params.size(); ++c) {
            std::string prefix = "c" + std::to_string(c + 1) + "_";
            
            model.params[c].vrad  = init_guess[prefix + "vrad"]["value"].get<double>();
            model.params[c].vsini = init_guess[prefix + "vsini"]["value"].get<double>();
            model.params[c].zeta  = init_guess[prefix + "zeta"]["value"].get<double>();
            model.params[c].teff  = init_guess[prefix + "teff"]["value"].get<double>();
            model.params[c].logg  = init_guess[prefix + "logg"]["value"].get<double>();
            model.params[c].xi    = init_guess[prefix + "xi"]["value"].get<double>();
            model.params[c].z     = init_guess[prefix + "z"]["value"].get<double>();
            model.params[c].he    = init_guess[prefix + "HE"]["value"].get<double>();
            
            frozen_status[c]["vrad"]  = init_guess[prefix + "vrad"]["freeze"].get<bool>();
            frozen_status[c]["vsini"] = init_guess[prefix + "vsini"]["freeze"].get<bool>();
            frozen_status[c]["zeta"]  = init_guess[prefix + "zeta"]["freeze"].get<bool>();
            frozen_status[c]["teff"]  = init_guess[prefix + "teff"]["freeze"].get<bool>();
            frozen_status[c]["logg"]  = init_guess[prefix + "logg"]["freeze"].get<bool>();
            frozen_status[c]["xi"]    = init_guess[prefix + "xi"]["freeze"].get<bool>();
            frozen_status[c]["z"]     = init_guess[prefix + "z"]["freeze"].get<bool>();
            frozen_status[c]["he"]    = init_guess[prefix + "HE"]["freeze"].get<bool>();
        }
        
        // Handle --output-synthetic flag
        if (cli.count("output-synthetic")) {
            std::cout << "Generating synthetic spectrum without spectral degradation...\n";
            
            // Get output path from JSON
            std::string output_dir = fit_cfg["outputPath"].get<std::string>();
            fs::create_directories(output_dir);
            
            auto init_guess = fit_cfg["initialGuess"];
            
            // Generate synthetic for each component
            for (size_t c = 0; c < model.params.size(); ++c) {
                std::string prefix = "c" + std::to_string(c + 1) + "_";
                
                // Read atmospheric parameters from initial guess
                StellarParams sp;
                sp.teff  = init_guess[prefix + "teff"]["value"].get<double>();
                sp.logg  = init_guess[prefix + "logg"]["value"].get<double>();
                sp.xi    = init_guess[prefix + "xi"]["value"].get<double>();
                sp.z     = init_guess[prefix + "z"]["value"].get<double>();
                sp.he    = init_guess[prefix + "HE"]["value"].get<double>();
                
                // These are set to zero for pure spectrum (no kinematic effects)
                sp.vrad  = 0.0;
                sp.vsini = 0.0;
                sp.zeta  = 0.0;
                
                std::cout << "\nComponent " << (c+1) << " parameters:\n"
                        << "  Teff  = " << sp.teff << " K\n"
                        << "  log g = " << sp.logg << "\n"
                        << "  [M/H] = " << sp.z << "\n"
                        << "  He    = " << sp.he << "\n"
                        << "  xi    = " << sp.xi << " km/s\n";
                
                // Compute the pure synthetic spectrum
                Spectrum synth = compute_synthetic_pure(model.grids[c], sp);
                
                // Build output filename
                std::ostringstream fname;
                fname << output_dir << "/synthetic.dat";
                std::string outfile = fname.str();
                
                // Write to file
                std::ofstream ofs(outfile);
                if (!ofs) {
                    throw std::runtime_error("Cannot write to " + outfile);
                }
                
                ofs << "# Synthetic spectrum (normalized, no spectral degradation)\n"
                    << "# Generated by DIGGA --output-synthetic\n"
                    << "#\n"
                    << "# Atmospheric parameters:\n"
                    << "#   Teff  = " << sp.teff << " K\n"
                    << "#   log g = " << sp.logg << " [cgs]\n"
                    << "#   [M/H] = " << sp.z << " dex\n"
                    << "#   He    = " << sp.he << "\n"
                    << "#   xi    = " << sp.xi << " km/s\n"
                    << "#\n"
                    << "# No rotational broadening (vsini = 0)\n"
                    << "# No macroturbulence (zeta = 0)\n"
                    << "# No radial velocity shift (vrad = 0)\n"
                    << "# No instrumental resolution degradation\n"
                    << "#\n"
                    << "# Columns: wavelength[Angstrom]  normalized_flux\n";
                
                ofs << std::scientific << std::setprecision(8);
                for (int i = 0; i < synth.lambda.size(); ++i) {
                    ofs << synth.lambda[i] << "  " << synth.flux[i] << "\n";
                }
                ofs.close();
                
                std::cout << "Written: " << outfile << " (" << synth.lambda.size() << " points)\n";
            }
            
            std::cout << "\nSynthetic spectrum output completed.\n";
            return 0;  // Exit without running the full fit
        }

        // Load all datasets
        std::vector<DataSet> datasets;

        // Get SNR filter threshold (0 means no filtering)
        double filter_snr = 0.0;
        if (global_cfg["settings"].contains("filter_snr")) {
            filter_snr = global_cfg["settings"]["filter_snr"].get<double>();
        }

        // Get minimum wavelength requirement (optional)
        double require_blue = -std::numeric_limits<double>::infinity();
        if (global_cfg["settings"].contains("requireBlue")) {
            require_blue = global_cfg["settings"]["requireBlue"].get<double>();
        }

        // Track statistics
        int total_spectra = 0;
        int rejected_spectra = 0;
        std::vector<std::string> rejected_files;

        if (filter_snr > 0) {
            std::cout << "Filtering spectra with SNR < " << filter_snr << "\n";
        }
        if (require_blue > 0) {
            std::cout << "Requiring minimum wavelength < " << require_blue << " Å\n";
        }

        for (const auto& obs : fit_cfg["observations"]) {
            for (const auto& file_cfg : obs["files"]) {
                total_spectra++;
                
                // Load spectrum
                std::string fpath = file_cfg["filename"].get<std::string>();
                std::string format = file_cfg["spectype"].get<std::string>();

                Spectrum raw;
                bool load_failed = false;
                
                try {
                    raw = load_spectrum(fpath, format);
                }
                catch (const std::exception& ex) {
                    std::cerr << "Failed to read spectrum: " << ex.what() << '\n';
                    load_failed = true;
                }

                // Check if spectrum should be rejected
                bool reject_spectrum = false;
                double snr_median = 0.0;
                double wmin = std::numeric_limits<double>::infinity();
                
                if (!load_failed && raw.lambda.size() > 0) {
                    // Compute minimum wavelength
                    wmin = raw.lambda.minCoeff();
                    
                    // Compute median SNR if filtering is enabled
                    if (filter_snr > 0 || require_blue > -std::numeric_limits<double>::infinity()) {
                        try {
                            // Use DER_SNR method for overall SNR estimate
                            SNRResult snr_result = raw.estimate_snr_der();
                            snr_median = snr_result.snr;
                            
                            // Alternative: use windowed SNR and take median
                            // SNRCurve curve = raw.estimate_snr_curve("der_snr", 300);
                            // snr_median = median(curve.snr);
                        }
                        catch (const std::exception& ex) {
                            std::cerr << "Warning: SNR estimation failed for " 
                                    << fs::path(fpath).filename() << ": " << ex.what() << '\n';
                            snr_median = 0.0;
                        }
                    }

                    // Check rejection criteria
                    if (filter_snr > 0 && snr_median <= filter_snr) {
                        //std::cout << "SNR " << snr_median << " filter_snr " << filter_snr << std::endl;
                        reject_spectrum = true;
                    }
                    if (require_blue > 0 && wmin >= require_blue) {
                        //std::cout << "Min Lambda " << wmin << " Require_blue " << require_blue << std::endl;
                        reject_spectrum = true;
                    }
                } else {
                    // Failed to load or empty spectrum
                    reject_spectrum = true;
                    snr_median = 0.0;
                    wmin = std::numeric_limits<double>::quiet_NaN();
                }
                
                // Report and skip rejected spectra
                if (reject_spectrum) {
                    rejected_spectra++;
                    rejected_files.push_back(fpath);
                    
                    std::cout << "Ignoring spectrum " << fs::path(fpath).filename() 
                            << " (SNR=" << std::fixed << std::setprecision(1) << snr_median 
                            << ", λ_min=" << std::fixed << std::setprecision(0) << wmin 
                            << ")\n";
                    continue;  // Skip this spectrum
                }

                /* ------------------------------------------------------------
                *  Per-file settings (with fall-back to the "observation" level)
                * ------------------------------------------------------------ */
                double res_offset = file_cfg["resOffset"].get<double>();
                double res_slope  = file_cfg["resSlope" ].get<double>();

                Vector nyquist = build_nyquist_grid(
                                raw.lambda.minCoeff(), raw.lambda.maxCoeff(),
                                res_offset, res_slope);
                            
                Spectrum rebinned;
                rebinned.lambda = nyquist;
                rebinned.flux = trapezoidal_rebin(raw.lambda, raw.flux, nyquist);
                rebinned.sigma = trapezoidal_rebin(raw.lambda, raw.sigma, nyquist);

                assert(std::is_sorted(rebinned.lambda.data(),
                    rebinned.lambda.data() + rebinned.lambda.size()) &&
                    "rebinned.lambda must be sorted (ascending)");

                std::vector<int> flags(rebinned.lambda.size(), 1);

                /* ------------------------------------------------------------
                *  Per-file settings with fall-back to the observation level
                * ------------------------------------------------------------ */

                /* --- wave-cut ------------------------------------------------ */
                std::array<double,2> wave_cut;
                if (file_cfg.contains("waveCut"))
                    wave_cut = file_cfg["waveCut"].get<std::array<double,2>>();
                else if (obs.contains("waveCut"))
                    wave_cut = obs["waveCut"].get<std::array<double,2>>();
                else   // no wave-cut given → keep everything
                    wave_cut = { -std::numeric_limits<double>::infinity(),
                                  +std::numeric_limits<double>::infinity() };

                /* --- ignore ranges ------------------------------------------ */
                std::vector<std::array<double,2>> ignore_ranges;
                if (file_cfg.contains("ignore"))
                    ignore_ranges = file_cfg["ignore"].get<std::vector<std::array<double,2>>>();
                else if (obs.contains("ignore"))
                    ignore_ranges = obs["ignore"].get<std::vector<std::array<double,2>>>();

                /* --- flag pixels to be rejected / kept ---------------------- */
                for (size_t j = 0; j < rebinned.lambda.size(); ++j) {
                    const double wl = rebinned.lambda[j];

                    if (wl < wave_cut[0] || wl > wave_cut[1])
                        flags[j] = 0;

                    for (const auto &rng : ignore_ranges)
                        if (wl >= rng[0] && wl <= rng[1]) {
                            flags[j] = 0;
                            break;
                        }
                }

                rebinned.ignoreflag = flags;

                /* --- cspline anchor-points ---------------------------------- */
                nlohmann::json anchor_json;                // copy, not reference!
                if (file_cfg.contains("csplineAnchorpoints"))
                    anchor_json = file_cfg["csplineAnchorpoints"];
                else if (obs.contains("csplineAnchorpoints"))
                    anchor_json = obs["csplineAnchorpoints"];
                else
                    anchor_json = nlohmann::json::array();

                std::vector<std::tuple<double,double,double>> cont_intervals;
                for (const auto &arr : anchor_json) {
                    cont_intervals.emplace_back(arr[0].get<double>(),
                                                arr[1].get<double>(),
                                                arr[2].get<double>());
                }

                
                Vector cont_x = anchors_from_intervals(cont_intervals, rebinned);
                Vector flux_at_anchors = interp_linear(rebinned.lambda, rebinned.flux, cont_x);
                
                // Create dataset
                DataSet ds;
                ds.name = fpath;
                ds.obs = rebinned;
                ds.cont_x = cont_x;
                ds.cont_y.resize(cont_x.size());
                for (int i = 0; i < cont_x.size(); ++i) {
                    ds.cont_y[i] = std::max(flux_at_anchors[i], 1e-6);
                }
                ds.resOffset = res_offset;
                ds.resSlope = res_slope;
                ds.keep = flags;
                
                datasets.push_back(ds);
            }
        }

        // Final check: ensure at least one spectrum passed the filters
        if (datasets.empty()) {
            std::ostringstream errstr;
            errstr << "No spectra passed the quality filters";
            if (filter_snr > 0) {
                errstr << " [SNR>" << filter_snr << "]";
            }
            if (require_blue > -std::numeric_limits<double>::infinity()) {
                errstr << " [λ_min<" << require_blue << "]";
            }
            throw std::runtime_error(errstr.str());
        }
        
        UnifiedFitWorkflow::Config workflow_config;

        /* ------------------------------------------------------------------ */
        /* generic flags the CLI already sets                                  */
        workflow_config.verbose     = true;
        workflow_config.debug_plots = cli.count("debug-plots") > 0;
        const bool make_plots = cli.count("no-plots")==0;   // true → produce plots
        const bool make_pdf   = cli.count("no-pdf"  )==0;   // true → run pdflatex

        /* ------------------------------------------------------------------ */
        /*  read optional tuning parameters from global_config.json            */
        if (global_cfg["settings"].contains("untieParams"))
            workflow_config.untie_params =
                global_cfg["settings"]["untieParams"]
                          .get<std::vector<std::string>>();

        /* ---- iterative noise -------------------------------------------- */
        auto &cfg_json = global_cfg["settings"];
        #define READ_OPT_INT(key, dst)   if (cfg_json.contains(key)) workflow_config.dst = cfg_json[key].get<int>();
        #define READ_OPT_DBL(key, dst)   if (cfg_json.contains(key)) workflow_config.dst = cfg_json[key].get<double>();

        READ_OPT_INT ("nitNoiseMax" ,  nit_noise_max );
        READ_OPT_INT ("nitFitMax"   ,  nit_fit_max   );
        READ_OPT_INT ("widthBoxPx"  ,  width_box_px  );
        READ_OPT_DBL ("outlierSigmaLo", outlier_sigma_lo );
        READ_OPT_DBL ("outlierSigmaHi", outlier_sigma_hi );
        READ_OPT_DBL ("convRangeLo" ,  conv_range_lo );
        READ_OPT_DBL ("convRangeHi" ,  conv_range_hi );
        READ_OPT_DBL ("convFraction",  conv_fraction );

        /* ------------------------------------------------------------------ */
        UnifiedFitWorkflow workflow(datasets, model, workflow_config,
                                    frozen_status, nthreads);
        workflow.run();
        
        /* ------------------------------------------------------------- */
        /*          create parameter table + summary plots               */
        /* ------------------------------------------------------------- */
        double xrange = global_cfg["settings"]["xrange"].get<double>();

        /* list of untied parameters (may be empty) -----------------------*/
        std::vector<std::string> untied_params;
        if (global_cfg["settings"].contains("untieParams"))
            untied_params =
                global_cfg["settings"]["untieParams"].get<std::vector<std::string>>();
        
        generate_results(fit_cfg["outputPath"].get<std::string>(),
                         workflow,
                         datasets,
                         model,
                         xrange,
                         /*grey=*/true,
                         untied_params,
                         make_plots, 
                         make_pdf); 

        std::cout << "\nFit completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    int hours = duration / 3600;
    int minutes = (duration % 3600) / 60;
    int seconds = duration % 60;

    std::cout << "\nTook: ";
    if (hours > 0) std::cout << hours << "h ";
    if (minutes > 0 || hours > 0) std::cout << minutes << "m ";
    std::cout << seconds << "s\n";
    
    return 0;
}
