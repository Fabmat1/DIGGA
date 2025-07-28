#include "specfit/JsonUtils.hpp"
#include "specfit/UnifiedFitWorkflow.hpp"
#include "specfit/ContinuumUtils.hpp"
#include "specfit/NyquistGrid.hpp"
#include "specfit/Rebin.hpp"
#include "specfit/ReportUtils.hpp"
#include "specfit/CommonTypes.hpp"
#include "specfit/SpectrumLoaders.hpp"
#include "specfit/SpectrumCache.hpp"
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
            ("cache-size", "Maximum number of cache entries", cxxopts::value<int>()->default_value("100"))
            ("debug-plots", "Enable debug plotting output")
            ("h,help", "Show help");
        
        auto cli = opts.parse(argc, argv);
        if (cli.count("help") || !cli.count("fit")) {
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
        
        // Load all datasets
        std::vector<DataSet> datasets;
        for (const auto& obs : fit_cfg["observations"]) {
            for (const auto& file_cfg : obs["files"]) {
                // Load spectrum
                std::string fpath = file_cfg["filename"].get<std::string>();
                std::string format = file_cfg["spectype"].get<std::string>();
                Spectrum raw;
                try {
                    raw = load_spectrum(fpath, format);
                }
                catch (const std::exception& ex) {
                    std::cerr << "Failed to read spectrum: " << ex.what() << '\n';
                }
                
                // Build Nyquist grid and rebin
                double res_offset = file_cfg["resOffset"].get<double>();
                double res_slope = file_cfg["resSlope"].get<double>();
                
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
                auto wave_cut = obs["waveCut"].get<std::array<double, 2>>();
                for (size_t j = 0; j < rebinned.lambda.size(); ++j) {
                    // Apply cuts and masks
                    const double wl = rebinned.lambda[j];

                    if (wl < wave_cut[0] || wl > wave_cut[1])
                        flags[j] = 0;

                    for (auto rng : obs["ignore"].get<std::vector<std::array<double,2>>>())
                        if (wl >= rng[0] && wl <= rng[1])
                            flags[j] = 0;
                }

                rebinned.ignoreflag = flags;
                
                // Setup continuum
                std::vector<std::tuple<double, double, double>> cont_intervals;
                for (auto arr : obs["csplineAnchorpoints"]) {
                    cont_intervals.emplace_back(
                        arr[0].get<double>(),
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
                
                std::cout << "Loaded: " << fs::path(fpath).filename() 
                          << " (" << rebinned.lambda.size() << " points)\n";
            }
        }
        
        // Run unified workflow
        UnifiedFitWorkflow::Config workflow_config;
        workflow_config.verbose = true;
        workflow_config.debug_plots = cli.count("debug-plots") > 0;
        
        /* read untied parameter list (optional) ---------------------- */
        if (global_cfg["settings"].contains("untieParams"))
            workflow_config.untie_params =
                global_cfg["settings"]["untieParams"]
                          .get<std::vector<std::string>>();

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
                         untied_params);

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
