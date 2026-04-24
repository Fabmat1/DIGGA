//  specfit_chi2.cpp
//  -----------------------------------------------------------
//  Post–processing tool:  read   fit_parameters.csv
//                         read   fit_report.tex   (only for the
//                                                   “spec ↔ dataset” map)
//                         read   original fit_config.json
//                         →  rebuild Model + DataSet
//                         →  compute reduced χ² for every spectrum
//                         →  print “<file>  <chi2_red>”
//
//  compile with the same include / link options that are used
//  for the main  specfit  executable.
//
#include "specfit/JsonUtils.hpp"
#include "specfit/ContinuumUtils.hpp"
#include "specfit/Rebin.hpp"
#include "specfit/NyquistGrid.hpp"
#include "specfit/SpectrumLoaders.hpp"
#include "specfit/SpectrumCache.hpp"
#include "specfit/CommonTypes.hpp"
#include "specfit/SyntheticModel.hpp"
#include "specfit/AkimaSpline.hpp"
#include "specfit/ModelGrid.hpp"

#include <cxxopts.hpp>
#include <Eigen/Core>
#include <gnuplot-iostream.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>

using namespace specfit;
namespace fs = std::filesystem;

/* ------------------------------------------------------------------------- */
/*  helper: load global_settings.json  (identical to the main executable)     */
/* ------------------------------------------------------------------------- */
static nlohmann::json load_global_config()
{
    std::vector<std::string> search_paths = {
        "global_settings.json",
        []{
            try {
                auto exe = fs::canonical("/proc/self/exe");
                return (exe.parent_path()/"global_settings.json").string();
            } catch (...) { return std::string("./global_settings.json"); }
        }(),
        []{
            try {
                auto exe = fs::read_symlink("/proc/self/exe");
                return (exe.parent_path()/"global_settings.json").string();
            } catch (...) { return std::string("./global_settings.json"); }
        }(),
        "../global_settings.json",
        "../../global_settings.json"
    };

    for (const auto& p: search_paths)
        if (fs::exists(p))
        {
            std::ifstream f(p);
            if (f) {
                nlohmann::json cfg;  f >> cfg;
                return cfg;
            }
        }
    throw std::runtime_error("cannot find global_settings.json");
}

/* ------------------------------------------------------------------------- */
/*  CSV parsing:  parameter,value,error  →  map                              */
/* ------------------------------------------------------------------------- */
struct ParamEntry {
    double value  = 0.;
    double error  = 0.;            // ==0   → parameter was frozen
};
using ParamTable = std::unordered_map<std::string,ParamEntry>;

static ParamTable read_csv(const std::string& fname)
{
    std::ifstream in(fname);
    if(!in) throw std::runtime_error("cannot read "+fname);

    ParamTable tab;
    std::string line;                       // skip header
    std::getline(in,line);

    while (std::getline(in,line))
    {
        std::istringstream ls(line);
        std::string key, val, err;
        std::getline(ls,key, ',');
        std::getline(ls,val, ',');
        std::getline(ls,err, ',');
        if (key.empty()) continue;

        ParamEntry e;
        e.value = std::stod(val);
        e.error = std::stod(err);
        tab.emplace(std::move(key), e);
    }
    return tab;
}

/* ------------------------------------------------------------------------- */
/*  TEX report:  we only need the mapping  spec# → name                      */
/* ------------------------------------------------------------------------- */
static std::vector<std::string>
read_tex_specnames(const std::string& texfile)
{
    std::ifstream in(texfile);
    if(!in) throw std::runtime_error("cannot read "+texfile);

    // lines look like:
    //  spec 2 & \verb|20150116_EG024027N133137V01_02_223_01.txt|.
    const std::regex pattern(R"(spec\s+(\d+)\s*&\s*\\verb\|([^|]+)\|)");
    std::vector<std::string> names;
    std::string line;
    while (std::getline(in,line))
    {
        std::smatch m;
        if (std::regex_search(line,m,pattern))
        {
            int idx = std::stoi(m[1].str());
            const std::string file = m[2].str();
            if (idx>0) {
                if (names.size()<static_cast<std::size_t>(idx))
                    names.resize(idx);
                names[idx-1] = file;
            }
        }
    }
    return names;
}

/* ------------------------------------------------------------------------- */
/*  grab a value from the ParamTable  (dataset-aware variants are tried       */
/*  first, then the global “c1_teff”, …)                                      */
/* ------------------------------------------------------------------------- */
static bool param_exists(const ParamTable& T, const std::string& key)
{ return T.find(key)!=T.end(); }

static ParamEntry get_param(const ParamTable& T,
                            const std::string& key,
                            const ParamEntry& def = {})
{
    const auto it = T.find(key);
    return (it==T.end()) ? def : it->second;
}

/* ------------------------------------------------------------------------- */
/*  command line, reconstruction, χ²                                         */
/* ------------------------------------------------------------------------- */
int main(int argc,char* argv[])
try
{
    /* ==================   CLI   ======================================= */
    cxxopts::Options opts("specfit_chi2",
                          "Compute per-spectrum reduced χ² from a finished fit");
    opts.add_options()
    ("csv",  "fit_parameters.csv", cxxopts::value<std::string>())
    ("tex",  "fit_report.tex",     cxxopts::value<std::string>())
    ("fit",  "fit_config.json",    cxxopts::value<std::string>())
    ("threads", "number of threads",
                cxxopts::value<int>()->default_value("0"))
    ("threshold", "only use points where |1-model| > threshold for chi2",
                cxxopts::value<double>()->default_value("0.0"))
    ("test", "plot first spectrum with gnuplot", cxxopts::value<bool>()->default_value("false"))
    ("h,help","show help");
    auto cli = opts.parse(argc,argv);
    if (cli.count("help")||!cli.count("csv")||!cli.count("fit")||!cli.count("tex"))
    {  std::cout<<opts.help()<<'\n'; return 0;  }

    /* ==================   env / thread pool   ========================= */
    int nth = cli["threads"].as<int>();
    if (nth<=0) nth = std::thread::hardware_concurrency();
    Eigen::setNbThreads(nth);

    double line_threshold = cli["threshold"].as<double>();

    /* ==================   read 3 input files   ======================== */
    auto global_cfg  = load_global_config();
    expand_env(global_cfg);          // ← NEW  (exactly the same as in main.cpp)
    auto       fit_cfg     = load_json(cli["fit"].as<std::string>());
    expand_env(fit_cfg);
    const auto csv_table   = read_csv(cli["csv"].as<std::string>());
    const auto spec_names  = read_tex_specnames(cli["tex"].as<std::string>());

    expand_env(fit_cfg);

    /* ==================   build Model  ================================ */
    SharedModel model;
    const auto& basePaths = global_cfg["basePaths"]
                                   .get<std::vector<std::string>>();
    for (const auto& g : fit_cfg["grids"])
        model.grids.emplace_back(basePaths,g);

    const unsigned n_components =
        static_cast<unsigned>(model.grids.size());     // = #components
    model.params.resize(n_components);                 // allocate structs

    /* ==================   load ALL spectra (as in the fit)  =========== */
    std::vector<DataSet> datasets;
    datasets.reserve(spec_names.size());

    unsigned dataset_index = 0;               // running index (d1,d2,…)

    for (const auto& obs  : fit_cfg["observations"])
        for (const auto& file_cfg : obs["files"])
        {
            /* ---- load original spectrum ----------------------------- */
            const std::string fname  = file_cfg["filename"];
            const std::string format = file_cfg["spectype"];
            Spectrum raw = load_spectrum(fname,format);

            /* ---- build Nyquist grid and re-bin ---------------------- */
            const double resOff = file_cfg["resOffset"];
            const double resSlp = file_cfg["resSlope"];
            Vector ny = build_nyquist_grid(raw.lambda.minCoeff(),
                                            raw.lambda.maxCoeff(),
                                            resOff,resSlp);

            DataSet ds;
            ds.name      = fname;
            ds.obs.lambda= ny;
            ds.obs.flux  = trapezoidal_rebin(raw.lambda,raw.flux ,ny);
            ds.obs.sigma = trapezoidal_rebin(raw.lambda,raw.sigma,ny);
            ds.resOffset = resOff;
            ds.resSlope  = resSlp;
            ds.keep.assign(ny.size(),1);
            ds.obs.ignoreflag = ds.keep;   // identical meaning

            /* ---- waveCut + ignore (just like the fit) --------------- */
            auto get_wavecut = [&]{
                if (file_cfg.contains("waveCut"))
                    return file_cfg["waveCut"]
                            .get<std::array<double,2>>();
                if (obs.contains("waveCut"))
                    return obs["waveCut"]
                            .get<std::array<double,2>>();
                return std::array<double,2>{-1e99,1e99};
            };
            const auto wc = get_wavecut();

            std::vector<std::array<double,2>> ignore;
            if (file_cfg.contains("ignore")) ignore =
                file_cfg["ignore"].get<std::vector<std::array<double,2>>>();
            else if (obs.contains("ignore")) ignore =
                obs["ignore"].get<std::vector<std::array<double,2>>>();

            for (int i=0;i<ny.size();++i)
            {
                double wl = ny[i];
                if (wl<wc[0]||wl>wc[1]) ds.obs.ignoreflag[i]=0;
                for (auto& r:ignore)
                    if (wl>=r[0]&&wl<=r[1]) {ds.obs.ignoreflag[i]=0;break;}
            }

            /* ---- continuum anchors from CSV ------------------------- */
            const std::string stem = fs::path(fname).stem().string();  // eg  …_00
            
            if (csv_table.find(stem + "_cont0") == csv_table.end())
            {
                std::cout << "Skipping   " << stem << "  (not in fit)\n";
                continue;                     // ← do NOT create a DataSet
            }
            
            std::vector<double> cx,cy;
            for (int k=0;;++k)
            {
                std::ostringstream keyx, keyy;
                keyx<<stem<<"_contX"<<k;
                keyy<<stem<<"_cont"<<k;
                if (!param_exists(csv_table,keyy.str())) break;   // Y is mandatory
                cy.push_back( get_param(csv_table,keyy.str()).value );
                double x = get_param(csv_table,keyx.str()).value; // exists – CSV has X list
                cx.push_back(x);
            }
            ds.cont_x = Eigen::Map<Vector>(cx.data(), cx.size());                
            ds.cont_y = cy;

            datasets.push_back(std::move(ds));
            ++dataset_index;
        }

    const unsigned NDS = static_cast<unsigned>(datasets.size());
    if (NDS==0) throw std::runtime_error("no spectra reconstructed");

    /* ==================   fill model / per-dataset stellar params ===== */
    struct PerCompPerDS {
        StellarParams  st;          // vrad,vsini,… for that component+ds
    };
    std::vector< std::vector<PerCompPerDS> > S(n_components,
                                               std::vector<PerCompPerDS>(NDS));

    auto grab_param = [&](unsigned c,const std::string& pname,
                          unsigned d)->double
    {
        std::ostringstream key_ds,key_gl;
        key_ds<<"c"<<(c+1)<<"_"<<pname<<"_d"<<(d+1);
        key_gl<<"c"<<(c+1)<<"_"<<pname;
        if (param_exists(csv_table,key_ds.str()))
            return get_param(csv_table,key_ds.str()).value;
        if (param_exists(csv_table,key_gl.str()))
            return get_param(csv_table,key_gl.str()).value;
        return 0.0;                      /* fallback */
    };

    for (unsigned c=0;c<n_components;++c)
    {
        for (unsigned d=0; d<NDS; ++d)
        {
            auto& p = S[c][d].st;
            p.vrad  = grab_param(c,"vrad",d);
            p.vsini = grab_param(c,"vsini",d);
            p.zeta  = grab_param(c,"zeta",d);
            p.teff  = grab_param(c,"teff",d);
            p.logg  = grab_param(c,"logg",d);
            p.xi    = grab_param(c,"xi",d);
            p.z     = grab_param(c,"z",d);
            p.he    = grab_param(c,"he",d);
        }
        /* store the parameters of the first dataset in model.params[]
           (that is exactly what the original workflow does at the end)      */
        model.params[c] = S[c][0].st;
    }

    /* ==================   helper: build synthetic + continuum  ========= */
    auto make_model = [&](unsigned d, Vector* mdl_no_cont = nullptr)->Vector
    {
        const DataSet& ds = datasets[d];
        const int npix = ds.obs.lambda.size();

        Vector mdl = Vector::Zero(npix);
        double wsum = 0.0;

        for (unsigned c=0;c<n_components;++c)
        {
            const auto& sp = S[c][d].st;
            Spectrum synth = compute_synthetic(model.grids[c], sp,
                                            ds.obs.lambda,
                                            ds.resOffset, ds.resSlope);
            double w = std::pow(sp.teff,4);
            mdl += w * synth.flux;
            wsum += w;
        }
        if (wsum>0) mdl /= wsum;

        // Store the continuum-free model if requested
        if (mdl_no_cont != nullptr) {
            *mdl_no_cont = mdl;
        }

        /* continuum */
        Vector vx = Eigen::Map<const Vector>(ds.cont_x.data(),
                                            ds.cont_x.size());
        Vector vy = Eigen::Map<const Vector>(ds.cont_y.data(),
                                            ds.cont_y.size());
        AkimaSpline spl(vx, vy);
        Vector cont = spl(ds.obs.lambda);
        mdl = mdl.cwiseProduct(cont);
        return mdl;
    };

    /* ==================   DOF counter (free parameters)  =============== */
    auto count_free_params = [&](unsigned d)->int
    {
        int free = 0;
        /* -- continuum anchors, one per dataset ------------------------- */
        const std::string stem = fs::path(datasets[d].name).stem().string();

        for (int k=0;;++k) {
            std::ostringstream ky;
            ky<<stem<<"_cont"<<k;
            auto it = csv_table.find(ky.str());
            if (it==csv_table.end()) break;
            if (it->second.error>0) ++free;
        }
        /* -- stellar params --------------------------------------------- */
        const char* pname[8]={"vrad","vsini","zeta","teff",
                              "logg","xi","z","he"};
        for (unsigned c=0;c<n_components;++c)
            for (int p=0;p<8;++p)
            {
                std::ostringstream key_ds,key_gl;
                key_ds<<"c"<<(c+1)<<"_"<<pname[p]<<"_d"<<(d+1);
                key_gl<<"c"<<(c+1)<<"_"<<pname[p];
                const auto it1 = csv_table.find(key_ds.str());
                if (it1!=csv_table.end() && it1->second.error>0) {++free;continue;}
                const auto it2 = csv_table.find(key_gl.str());
                if (it2!=csv_table.end() && it2->second.error>0)  {
                    /* global parameter counts only once: add it in d==0 */
                    if (d==0) ++free;
                }
            }
        return free;
    };

    /* ==================   plotting helper  ================================= */
    auto plot_spectrum = [&](unsigned d, const Vector& mdl, const Vector& mdl_no_cont,
                            double line_threshold)
    {
        const DataSet& ds = datasets[d];
        Gnuplot gp;
        
        // Prepare data for plotting
        std::vector<std::tuple<double,double,double,double,int>> plot_data;
        for (int i=0; i<ds.obs.lambda.size(); ++i) {
            bool use_point = ds.obs.ignoreflag[i];
            if (line_threshold > 0.0 && use_point) {
                double deviation = std::abs(1.0 - mdl_no_cont[i]);
                use_point = (deviation > line_threshold);
            }
            plot_data.emplace_back(
                ds.obs.lambda[i],
                ds.obs.flux[i],
                ds.obs.sigma[i],
                mdl[i],
                use_point ? 1 : 0
            );
        }
        
        // Split into used and ignored points
        std::vector<std::pair<double,double>> obs_used, obs_ignored;
        std::vector<std::tuple<double,double,double>> obs_used_err;
        std::vector<std::pair<double,double>> model_line;
        
        for (const auto& pt : plot_data) {
            double wl = std::get<0>(pt);
            double fl = std::get<1>(pt);
            double sg = std::get<2>(pt);
            double md = std::get<3>(pt);
            int used = std::get<4>(pt);
            
            model_line.emplace_back(wl, md);
            
            if (used) {
                obs_used.emplace_back(wl, fl);
                obs_used_err.emplace_back(wl, fl, sg);
            } else {
                obs_ignored.emplace_back(wl, fl);
            }
        }
        
        // Calculate chi-square for display
        Vector resid = (ds.obs.flux - mdl).cwiseQuotient(ds.obs.sigma);
        double sum = 0.0;
        int count = 0;
        for (int i=0; i<resid.size(); ++i) {
            if (ds.obs.ignoreflag[i] && std::isfinite(resid[i])) {
                sum += resid[i];
                ++count;
            }
        }
        double mean_resid = (count > 0) ? (sum / count) : 0.0;
        
        for (int i=0; i<resid.size(); ++i) {
            if (!std::isfinite(resid[i])) {
                resid[i] = mean_resid;
            }
        }
        
        double chi2 = 0.0;
        int nobs = 0;
        for (int i=0; i<resid.size(); ++i) {
            if (ds.obs.ignoreflag[i]) {
                bool use_point = true;
                if (line_threshold > 0.0) {
                    double deviation = std::abs(1.0 - mdl_no_cont[i]);
                    use_point = (deviation > line_threshold);
                }
                if (use_point) {
                    chi2 += resid[i]*resid[i];
                    ++nobs;
                }
            }
        }
        
        int nfree = count_free_params(d);
        int dof = std::max(1, nobs - nfree);
        double chi2_red = chi2 / dof;
        
        // Plot
        gp << "set title 'Spectrum: " << ds.name << "\\nχ² = " << chi2_red 
        << " (threshold = " << line_threshold << ")'\n";
        gp << "set xlabel 'Wavelength (Å)'\n";
        gp << "set ylabel 'Flux'\n";
        gp << "set grid\n";
        gp << "set key outside right top\n";
        
        gp << "plot '-' with lines lw 2 lc rgb 'red' title 'Model', "
        << "'-' with points pt 7 ps 0.5 lc rgb 'blue' title 'Obs (used in χ²)', "
        << "'-' with yerrorbars pt 7 ps 0.3 lc rgb 'blue' notitle, "
        << "'-' with points pt 7 ps 0.3 lc rgb 'gray' title 'Obs (ignored)'\n";
        
        gp.send1d(model_line);
        gp.send1d(obs_used);
        gp.send1d(obs_used_err);
        gp.send1d(obs_ignored);
        
        std::cout << "\nPress Enter to continue...";
        std::cin.get();
    };

    /* ==================   χ² loop  ==================================== */
    std::cout<<std::fixed<<std::setprecision(6);

    bool test_mode = cli["test"].as<bool>();

    if (test_mode && NDS > 0) {
        // Test mode: plot first spectrum
        std::cout << "TEST MODE: Plotting first spectrum\n";
        const DataSet& ds = datasets[0];
        Vector mdl_no_cont;
        Vector mdl = make_model(0, &mdl_no_cont);
        
        std::cout << "\nSpectrum: " << fs::path(ds.name).filename().string() << "\n";
        std::cout << "Points in spectrum: " << ds.obs.lambda.size() << "\n";
        std::cout << "Points kept (ignoreflag=1): " 
                << std::count(ds.obs.ignoreflag.begin(), ds.obs.ignoreflag.end(), 1) << "\n";
        
        if (line_threshold > 0.0) {
            int line_points = 0;
            for (int i=0; i<mdl_no_cont.size(); ++i) {
                if (ds.obs.ignoreflag[i]) {
                    double deviation = std::abs(1.0 - mdl_no_cont[i]);
                    if (deviation > line_threshold) {
                        ++line_points;
                    }
                }
            }
            std::cout << "Points passing line threshold: " << line_points << "\n";
        }
        
        plot_spectrum(0, mdl, mdl_no_cont, line_threshold);
        return 0;
    }

    // Normal mode: compute chi-square for all spectra
    for (unsigned d=0; d<NDS; ++d)
    {
        const DataSet& ds = datasets[d];
        Vector mdl_no_cont;
        Vector mdl = make_model(d, &mdl_no_cont);
        Vector resid = (ds.obs.flux - mdl).cwiseQuotient(ds.obs.sigma);
        
        // Calculate mean of valid residuals (excluding ignored points and Inf/NaN)
        double sum = 0.0;
        int count = 0;
        for (int i=0; i<resid.size(); ++i) {
            if (ds.obs.ignoreflag[i] && std::isfinite(resid[i])) {
                sum += resid[i];
                ++count;
            }
        }
        double mean_resid = (count > 0) ? (sum / count) : 0.0;
        
        // Replace Inf/NaN with mean
        for (int i=0; i<resid.size(); ++i) {
            if (!std::isfinite(resid[i])) {
                resid[i] = mean_resid;
            }
        }
        
        // Calculate chi2, optionally filtering by line threshold
        double chi2 = 0.0;
        int    nobs = 0;
        for (int i=0; i<resid.size(); ++i) {
            if (ds.obs.ignoreflag[i]) {
                // Check if this point passes the line threshold
                bool use_point = true;
                if (line_threshold > 0.0) {
                    double deviation = std::abs(1.0 - mdl_no_cont[i]);
                    use_point = (deviation > line_threshold);
                }
                
                if (use_point) {
                    chi2 += resid[i]*resid[i];
                    ++nobs;
                }
            }
        }
        
        double chi2_red = chi2;
        
        //int nfree = count_free_params(d);
        //int dof   = std::max(1, nobs - nfree);
        //double chi2_red = chi2 / dof;
        
        std::cout << fs::path(ds.name).filename().string()
                << "  " << chi2_red << '\n';
    }
    
    return 0;
}
/* ------------------------------------------------------------------------- */
catch (const std::exception& e)
{
    std::cerr<<"Error: "<<e.what()<<'\n';
    return 1;
}