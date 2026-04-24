#include "specfit/JsonUtils.hpp"
#include "specfit/UnifiedFitWorkflow.hpp"
#include "specfit/CommonTypes.hpp"
#include "specfit/SyntheticModel.hpp"
#include "specfit/SpectrumCache.hpp"
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>

#ifdef DIGGA_HAVE_REPORT
#include "specfit/ReportUtils.hpp"
#endif

namespace fs = std::filesystem;
using namespace specfit;

namespace {

std::string find_global_settings()
{
    std::vector<std::string> candidates = {
        "global_settings.json",
        "../global_settings.json",
        "../../global_settings.json",
    };
    try {
        auto exe = fs::canonical("/proc/self/exe").parent_path();
        candidates.insert(candidates.begin(),
                          (exe / "global_settings.json").string());
    } catch (...) {}
    for (const auto& p : candidates)
        if (fs::exists(p)) { std::cout << "Loaded config from: " << p << '\n'; return p; }
    throw std::runtime_error("global_settings.json not found");
}

int run_synthetic_only(const api::FitInput& fi, const api::GlobalSettings& gs)
{
    fs::create_directories(fi.output_path);
    for (std::size_t c = 0; c < fi.components.size(); ++c) {
        ModelGrid grid(gs.base_paths, fi.components[c].grid_relative_path);
        StellarParams sp{};
        const auto& ci = fi.components[c];
        sp.teff = ci.teff; sp.logg = ci.logg; sp.xi = ci.xi;
        sp.z    = ci.z;    sp.he   = ci.he;
        Spectrum synth = compute_synthetic_pure(grid, sp);
        std::string out = fi.output_path + "/synthetic.dat";
        std::ofstream ofs(out);
        ofs << std::scientific << std::setprecision(8);
        for (Eigen::Index i = 0; i < synth.lambda.size(); ++i)
            ofs << synth.lambda[i] << "  " << synth.flux[i] << '\n';
        std::cout << "Written: " << out << '\n';
    }
    return 0;
}

} // anonymous

int main(int argc, char** argv)
{
    auto t0 = std::chrono::steady_clock::now();
    try {
        cxxopts::Options opts("DIGGA", "Multi-dataset stellar spectrum fitting");
        opts.add_options()
            ("fit",              "Fit configuration JSON", cxxopts::value<std::string>())
            ("threads",          "Number of threads",      cxxopts::value<int>()->default_value("0"))
            ("output-synthetic", "Only write undegraded synthetic spectra")
            ("cache-size",       "Cache entries",          cxxopts::value<int>()->default_value("100"))
            ("debug-plots",      "Write per-stage debug plots")
            ("no-plots",         "Skip per-spectrum summary plots")
            ("no-pdf",           "Do not run pdflatex")
            ("h,help",           "Show help");
        auto cli = opts.parse(argc, argv);
        if (cli.count("help") || !cli.count("fit")) {
            std::cout << opts.help() << '\n'; return 0;
        }

        SpectrumCache::instance().set_capacity(cli["cache-size"].as<int>());

        auto gs = api::global_settings_from_json_file(find_global_settings());
        auto fi = api::fit_input_from_json_file(cli["fit"].as<std::string>());
        if (cli.count("debug-plots")) {
        #ifdef DIGGA_HAVE_REPORT
            gs.debug_plots = true;
            gs.on_stage_complete =
                [&](int stage_idx, const specfit::UnifiedFitWorkflow& wf) {
                    namespace fs = std::filesystem;
                    fs::create_directories("debug");
                    specfit::MultiPanelPlotter P(1.0, false);
                    // you'll need access to datasets here — see note below
                };
        #else
            std::cerr << "--debug-plots requires DIGGA_BUILD_REPORT=ON\n";
        #endif
        }

        if (cli.count("output-synthetic"))
            return run_synthetic_only(fi, gs);

        api::DiggaSession session;
        session.set_global_settings(gs);
        session.set_fit_input(fi);
        session.set_num_threads(cli["threads"].as<int>());
        session.set_log_callback(
            [](const std::string& line){ std::cout << line << '\n'; });

        auto result = session.run();

        // --- keep existing report output exactly as before ---------------
        //  We need the live workflow/datasets for generate_results().
        //  Simplest: re-run would be wasteful, so either:
        //    (a) expose workflow+datasets from the session, or
        //    (b) (recommended) write a small adapter that produces the
        //        report straight from FitResult in a future patch.
        //  For now: if ASTRA is the primary consumer, reports are emitted
        //  by ASTRA itself from FitResult. The CLI just prints a summary.

        std::cout << "\nFit completed successfully.\n"
                  << "  chi2            = " << result.final_chi2         << '\n'
                  << "  iterations      = " << result.iterations         << '\n'
                  << "  free parameters = " << result.n_free_parameters  << '\n'
                  << "  data points     = " << result.n_data_points      << '\n'
                  << "  spectra used    = " << result.spectra.size()     << '\n'
                  << "  spectra rejected= " << result.rejected_files.size() << '\n';

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    auto dur = std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - t0).count();
    std::cout << "Took: " << dur/3600 << "h " << (dur%3600)/60 << "m "
              << dur%60 << "s\n";
    return 0;
}
