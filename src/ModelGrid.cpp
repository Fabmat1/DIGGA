#include "specfit/ModelGrid.hpp"
#include "specfit/Resolution.hpp"
#include "specfit/SpectrumCache.hpp"
#include <CCfits/CCfits>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <cmath>        // pow, lround
#include <algorithm>    // lower_bound
#include <cstdint>      // uint64_t
#include <cstring>      // memcpy
#include <array>
#include <cstdio>

namespace fs = std::filesystem;

/* tiny helpers outside namespace ------------------------------------ */
static std::string fmt(double v, int prec)
{
    std::ostringstream s;
    s << std::fixed << std::setprecision(prec) << v;
    return s.str();
}

static double to_linear(const std::string& axis_name, double value)
{
    if (axis_name == "g" || axis_name == "HHE")
        return std::pow(10.0, value);
    return value;
}

/* ------------------------------------------------------------------ *
 * hashing helpers for (path,resOffset,resSlope)                      *
 * ------------------------------------------------------------------ */
static std::size_t hash_combine(std::size_t seed, std::size_t v) noexcept
{
    seed ^= v + 0x9E3779B97F4A7C15ULL + (seed << 6) + (seed >> 2);
    return seed;
}
static std::size_t hash_double(double x) noexcept
{
    std::uint64_t bits;
    static_assert(sizeof(bits) == sizeof(x));
    std::memcpy(&bits, &x, sizeof(bits));
    return std::hash<std::uint64_t>{}(bits);
}

/* ==================================================================== */
namespace specfit {

/* --------- small Eigen wrapper ------------------------------------- */
static Vector to_eigen(const std::vector<Real>& v)
{
    return Eigen::Map<const Vector>(v.data(), v.size());
}

/* --------- resolve grid base path ---------------------------------- */
static std::string
resolve_grid(const std::vector<std::string>& bases,
             const std::string& rel_path)
{
    for (const auto& b : bases) {
        fs::path candidate = fs::path(b) / rel_path;
        if (fs::exists(candidate / "grid.fits")) return candidate.string();
    }
    throw std::runtime_error("Grid '" + rel_path +
                             "' not found in any basePath.");
}

/* =========================  Ctors  ================================= */
ModelGrid::ModelGrid(const std::vector<std::string>& base_paths,
                     const std::string& rel_path)
    : base_(resolve_grid(base_paths, rel_path))
{
    try {
        CCfits::FITS f(base_ + "/grid.fits", CCfits::Read);
        CCfits::ExtHDU& ext = f.extension(1);
        const auto& cols    = ext.column();

        for (const auto& [name, col] : cols) {
            std::vector<Real> buf;
            if (col->varLength())               col->read(buf, 1);
            else if (col->repeat() == 1)        col->read(buf, 1, 1);
            else                                col->read(buf, 1);
            axes_.push_back(GridAxis{ name, to_eigen(buf) });
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Failed reading grid metadata in "
                                 + base_ + "/grid.fits : " + e.what());
    }
}

ModelGrid::ModelGrid(std::string abs_path)
    : base_(std::move(abs_path))
{}

/* ======== helpers common to both load_spectrum variants ============ */
static double param_for_axis(const std::string& n,
                             double teff, double logg,
                             double z,    double he,
                             double xi)
{
    if (n == "t")   return teff;
    if (n == "g")   return logg;
    if (n == "z")   return z;
    if (n == "HHE") return he;
    if (n == "x")   return xi;
    throw std::runtime_error("Unsupported axis: " + n);
}

/* ------------------------------------------------------------------ *
 * 1)  original high-resolution spectrum                              *
 * ------------------------------------------------------------------ */
Spectrum ModelGrid::load_spectrum(double teff,
                                  double logg,
                                  double z,
                                  double he,
                                  double xi) const
{
    /* --------------------------------------------------------------
       build the interpolation hyper-cube
       -------------------------------------------------------------- */

    struct Node { double teff, logg, z, he, xi, weight; };

    /* 2⁵ = 32 corner points is the absolute maximum we will need */
    std::array<Node, 32> nodes;
    nodes[0] = {0,0,0,0,0,1.0};
    int n_nodes = 1;                      // number of valid entries

    for (const auto& ax : axes_)
    {
        const double   p_val     = param_for_axis(ax.name,
                                                  teff,logg,z,he,xi);
        const Vector&  grid_axis = ax.values;

        /* ------ degenerate axis (only one value) ------------------ */
        if (grid_axis.size() == 1)
        {
            for (int i = 0; i < n_nodes; ++i)
            {
                Node& n = nodes[i];
                if      (ax.name=="t")   n.teff = grid_axis[0];
                else if (ax.name=="g")   n.logg = grid_axis[0];
                else if (ax.name=="z")   n.z    = grid_axis[0];
                else if (ax.name=="HHE") n.he   = grid_axis[0];
                else if (ax.name=="x")   n.xi   = grid_axis[0];
            }
            continue;
        }

        /* ------ locate bounding grid points ----------------------- */
        auto it = std::lower_bound(grid_axis.data(),
                                   grid_axis.data()+grid_axis.size(),
                                   p_val);

        int hi = (it == grid_axis.data()+grid_axis.size())
                  ? int(grid_axis.size()) - 1
                  : int(it - grid_axis.data());
        int lo = (hi == 0) ? 0 : hi - 1;

        const double p_lin  = to_linear(ax.name, p_val);
        const double lo_lin = to_linear(ax.name, grid_axis[lo]);
        const double hi_lin = to_linear(ax.name, grid_axis[hi]);

        const double alpha_hi = (hi_lin==lo_lin) ? 0.0
                           : (p_lin - lo_lin) / (hi_lin - lo_lin);
        const double alpha_lo = 1.0 - alpha_hi;

        /* ------ duplicate the current set of nodes ---------------- */

        /* copy existing nodes to the upper half of the array         */
        for (int i = n_nodes-1; i >= 0; --i)
            nodes[i + n_nodes] = nodes[i];

        /* patch the coordinates + weights in-place                   */
        for (int i = 0; i < n_nodes; ++i)
        {
            Node& a = nodes[i];           // low-side  node
            Node& b = nodes[i+n_nodes];   // high-side node

            if      (ax.name=="t")  { a.teff = grid_axis[lo];
                                      b.teff = grid_axis[hi]; }
            else if (ax.name=="g")  { a.logg = grid_axis[lo];
                                      b.logg = grid_axis[hi]; }
            else if (ax.name=="z")  { a.z    = grid_axis[lo];
                                      b.z    = grid_axis[hi]; }
            else if (ax.name=="HHE"){ a.he   = grid_axis[lo];
                                      b.he   = grid_axis[hi]; }
            else if (ax.name=="x")  { a.xi   = grid_axis[lo];
                                      b.xi   = grid_axis[hi]; }

            a.weight *= alpha_lo;
            b.weight *= alpha_hi;
        }
        n_nodes *= 2;                     // doubled by one dimension
    }

    /* --------------------------------------------------------------
       spectra fetch helper (per-call local cache)
       -------------------------------------------------------------- */

    struct CacheEntry { const Spectrum* sp = nullptr; bool ok = false; };

    using Key = std::uint64_t;            // 5 × 12-bit packed integer
    auto make_key = [](int t,int g,int z,int he,int x) -> Key
    {
        return  (Key)t         |
               ((Key)g  << 12) |
               ((Key)z  << 24) |
               ((Key)he << 36) |
               ((Key)x  << 48);
    };

    std::unordered_map<Key, CacheEntry> local_cache;
    local_cache.reserve(n_nodes);         // never rehashes

    auto fetch = [&](const Node& n) -> const Spectrum&
    {
        const int Ti = static_cast<int>(std::lround(n.teff));
        const int Gi = static_cast<int>(std::lround(n.logg * 100));
        const int Zi = static_cast<int>(std::lround(n.z    * 100));
        const int Hi = static_cast<int>(std::lround(n.he   * 1000));
        const int Xi = static_cast<int>(std::lround(n.xi   * 100));

        Key key = make_key(Ti,Gi,Zi,Hi,Xi);

        auto& e = local_cache[key];
        if (!e.ok)
        {
            if (auto* sp = SpectrumCache::instance().try_get(key))
            {
                e.sp = sp;                // global cache hit
            }
            else
            {
                char path[128];
                std::snprintf(path, sizeof(path),
                              "%s/HHE/Z%02d/HE%03d/X%02d/G%03d/T%d.fits",
                              base_.c_str(), Zi, Hi, Xi, Gi, Ti);

                e.sp = &SpectrumCache::instance()
                           .insert_if_absent(key,
                                 [&]{ return read_fits(path); });
            }
            e.ok = true;
        }
        return *e.sp;
    };

    /* --------------------------------------------------------------
       weighted sum of the corner spectra
       -------------------------------------------------------------- */

    Spectrum out;
    bool   first = true;
    double wsum  = 0.0;

    for (int i = 0; i < n_nodes; ++i)
    {
        const Node&     nd = nodes[i];
        const Spectrum& sp = fetch(nd);

        if (first) { out = sp; out.flux.setZero(); first = false; }
        out.flux += sp.flux * nd.weight;
        wsum     += nd.weight;
    }

    if (wsum > 0.0) out.flux.array() /= wsum;
    return out;
}

/* ------------------------------------------------------------------ *
 * 2)  NEW  — resolution-aware variant                                *
 * ------------------------------------------------------------------ */

Spectrum ModelGrid::load_spectrum(double teff,
                                  double logg,
                                  double z,
                                  double he,
                                  double xi,
                                  double resOffset,
                                  double resSlope) const
{
    /* --------------------------------------------------------------
       interpolation hyper-cube   (≤ 2⁵ = 32 corner nodes)
       -------------------------------------------------------------- */
    struct Node { double teff,logg,z,he,xi,weight; };

    std::array<Node,32> nodes;
    nodes[0]  = {0,0,0,0,0,1.0};
    int nNodes = 1;

    for (const auto& ax : axes_)
    {
        const double   p_val = param_for_axis(ax.name,
                                             teff,logg,z,he,xi);
        const Vector&  grid  = ax.values;

        /* ---- axis that is constant in the grid -------------------- */
        if (grid.size()==1)
        {
            for (int i=0;i<nNodes;++i)
            {
                Node& n = nodes[i];
                if      (ax.name=="t")   n.teff = grid[0];
                else if (ax.name=="g")   n.logg = grid[0];
                else if (ax.name=="z")   n.z    = grid[0];
                else if (ax.name=="HHE") n.he   = grid[0];
                else if (ax.name=="x")   n.xi   = grid[0];
            }
            continue;
        }

        /* ---- locate bounding indices ------------------------------ */
        auto  it = std::lower_bound(grid.data(),
                                    grid.data()+grid.size(),
                                    p_val);
        int hi = (it==grid.data()+grid.size())
                 ? int(grid.size())-1
                 : int(it-grid.data());
        int lo = (hi==0)?0:hi-1;

        const double p_lin  = to_linear(ax.name,p_val);
        const double lo_lin = to_linear(ax.name,grid[lo]);
        const double hi_lin = to_linear(ax.name,grid[hi]);

        const double alpha_hi = (hi_lin==lo_lin) ? 0.0
                                 : (p_lin-lo_lin)/(hi_lin-lo_lin);
        const double alpha_lo = 1.0-alpha_hi;

        /* ---- duplicate existing nodes along this axis ------------- */
        for (int i=nNodes-1;i>=0;--i)      // copy backwards
            nodes[i+nNodes] = nodes[i];

        for (int i=0;i<nNodes;++i)
        {
            Node& a = nodes[i];            // low  side
            Node& b = nodes[i+nNodes];     // high side

            if      (ax.name=="t")  { a.teff = grid[lo]; b.teff = grid[hi]; }
            else if (ax.name=="g")  { a.logg = grid[lo]; b.logg = grid[hi]; }
            else if (ax.name=="z")  { a.z    = grid[lo]; b.z    = grid[hi]; }
            else if (ax.name=="HHE"){ a.he   = grid[lo]; b.he   = grid[hi]; }
            else if (ax.name=="x")  { a.xi   = grid[lo]; b.xi   = grid[hi]; }

            a.weight *= alpha_lo;
            b.weight *= alpha_hi;
        }
        nNodes *= 2;
    }

    /* --------------------------------------------------------------
       fast cache key: pack five 12-bit indices into 64 bits
       -------------------------------------------------------------- */
    using Key = std::uint64_t;
    auto pack_axes = [](int t,int g,int z,int he,int x)->Key
    {
        return  (Key)t         |
               ((Key)g  <<12) |
               ((Key)z  <<24) |
               ((Key)he <<36) |
               ((Key)x  <<48);
    };

    /* --------------------------------------------------------------
       per-call local cache (≤ 32 entries)
       -------------------------------------------------------------- */
    struct CacheEntry { const Spectrum* sp=nullptr; bool ok=false; };
    std::unordered_map<Key,CacheEntry> local_cache;
    local_cache.reserve(nNodes);

    auto fetch = [&](const Node& n) -> const Spectrum&
    {
        /* ---- build packed key for the five axes ------------------- */
        const int Ti = int(std::lround(n.teff));
        const int Gi = int(std::lround(n.logg*100));   // unchanged hash
        const int Zi = int(std::lround(n.z   *100));
        const int Hi = int(std::lround(n.he  *1000));
        const int Xi = int(std::lround(n.xi  *100));

        Key key = pack_axes(Ti,Gi,Zi,Hi,Xi);

        /* ---- incorporate resolution parameters -------------------- */
        key = hash_combine(key, hash_double(resOffset));
        key = hash_combine(key, hash_double(resSlope));

        /* ---- lookup in the call-local cache ----------------------- */
        auto& e = local_cache[key];
        if (!e.ok)
        {
            /* ---- try the global SpectrumCache first --------------- */
            if (auto* sp = SpectrumCache::instance().try_get(key))
            {
                e.sp = sp;           // global hit
            }
            else
            {
                /* ---- MISS: build the exact file name -------------- */
                std::ostringstream oss;
                oss << base_ << "/HHE"
                    << "/Z"  << fmt(n.z,  2)
                    << "/HE" << fmt(n.he, 3)
                    << "/X"  << fmt(n.xi, 2)
                    << "/G"  << fmt(n.logg,3)
                    << "/T"  << static_cast<int>(std::lround(n.teff))
                    << ".fits";

                const std::string path = oss.str();

                /* read FITS, degrade resolution, store in global cache */
                e.sp = &SpectrumCache::instance()
                           .insert_if_absent(key,[&]
                           {
                               Spectrum hi = read_fits(path);
                               Spectrum lo = hi;
                               lo.flux = degrade_resolution(hi.lambda, hi.flux,
                                                            resOffset,resSlope);
                               return lo;   // move-returned
                           });
            }
            e.ok = true;
        }
        return *e.sp;
    };

    /* --------------------------------------------------------------
       weighted sum of the corner spectra
       -------------------------------------------------------------- */
    Spectrum out;
    bool   first = true;
    double wsum  = 0.0;

    for (int i=0;i<nNodes;++i)
    {
        const Node&     nd = nodes[i];
        const Spectrum& sp = fetch(nd);

        if (first){ out = sp; out.flux.setZero(); first = false; }
        out.flux += sp.flux * nd.weight;
        wsum     += nd.weight;
    }

    if (wsum>0.0) out.flux.array() /= wsum;
    return out;
}

/* ------------------------------------------------------------------ *
 * read_fits()   (unchanged except for namespace)                     *
 * ------------------------------------------------------------------ */
Spectrum ModelGrid::read_fits(const std::string& path) const
{
    using Vec = std::vector<Real>;
    static std::unordered_map<std::string, Vector> wave_cache;

    Vector lam;
    auto it = wave_cache.find(base_);
    if (it != wave_cache.end()) lam = it->second;
    else {
        fs::path lam_path = fs::path(base_) / "HHE" / "lambda.fits";
        if (fs::exists(lam_path)) {
            CCfits::FITS fl(lam_path.string(), CCfits::Read);
            CCfits::ExtHDU& ext_l = fl.extension(1);
            Vec tmp; ext_l.column("l").read(tmp, 1, ext_l.rows());
            lam = Eigen::Map<Vector>(tmp.data(), tmp.size());
            wave_cache.emplace(base_, lam);
        }
    }

    CCfits::FITS f(path, CCfits::Read);
    CCfits::ExtHDU& ext = f.extension(1);

    Vec fl; ext.column("f").read(fl, 1, ext.rows());
    if (lam.size() == 0) {
        Vec ltmp; ext.column("l").read(ltmp, 1, ext.rows());
        lam = Eigen::Map<Vector>(ltmp.data(), ltmp.size());
        wave_cache.emplace(base_, lam);
    }

    Spectrum sp;
    sp.lambda = lam;
    sp.flux   = Eigen::Map<Vector>(fl.data(), fl.size());
    sp.sigma  = Vector::Ones(sp.lambda.size());

    static std::unordered_map<std::string,bool> said;
    if (!said[path]) {
        said[path] = true;
        std::cout << "[grid] " << fs::path(path).filename().string()
                  << "  λ=[" << lam.minCoeff() << " … " << lam.maxCoeff() << "]"
                  << "  n="  << lam.size()
                  << "  flux-range=[" << sp.flux.minCoeff()
                  << " … " << sp.flux.maxCoeff() << "]\n";
    }
    return sp;
}

} // namespace specfit