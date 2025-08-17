#include "specfit/SpectrumLoaders.hpp"
#include "specfit/SNRHelpers.hpp"            // get_signal_to_noise_curve / der_snr_curve
#include <Eigen/Dense>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfit {
namespace {

Vector linear_interpolate(const Vector& x_new,
                          const Vector& x_old,
                          const Vector& y_old)
{
    if (x_old.size() != y_old.size() || x_old.size() < 2)
        throw std::invalid_argument("linear_interpolate: invalid input table.");

    Vector y_new(x_new.size());

    for (Eigen::Index i = 0; i < x_new.size(); ++i)
    {
        const double x = x_new[i];

        if (x <= x_old[0]) {
            y_new[i] = y_old[0];
            continue;
        }
        if (x >= x_old[x_old.size() - 1]) {
            y_new[i] = y_old[y_old.size() - 1];
            continue;
        }

        // binary search
        const auto* first = x_old.data();
        const auto* last  = x_old.data() + x_old.size();
        const auto* it    = std::upper_bound(first, last, x);
        const Eigen::Index hi = static_cast<Eigen::Index>(it - first);
        const Eigen::Index lo = hi - 1;

        const double t =
            (x - x_old[lo]) / (x_old[hi] - x_old[lo]);      // 0 … 1
        y_new[i] = (1.0 - t) * y_old[lo] + t * y_old[hi];
    }
    return y_new;
}

inline double median(Vector v)               // pass *by value*  → cheap copy
{
    const Eigen::Index n = v.size();
    if (n == 0)
        throw std::runtime_error("median(): empty vector");

    Eigen::Index k = n / 2;
    std::nth_element(v.data(), v.data() + k, v.data() + n);   // k-th element

    double m = v[k];
    if ((n & 1) == 0) {                                       // even length
        const double max_lo = *std::max_element(v.data(), v.data() + k);
        m = 0.5 * (m + max_lo);
    }
    return m;
}

// ----------------------------------------------------------------------------
//  Read an ASCII table with N columns of doubles, skip comment lines
// ----------------------------------------------------------------------------
std::vector<std::array<double, 3>>
read_ascii_table(const std::string& path,
                 int  ncols,                        // 2 or 3
                 char comment_char = '#')
{
    if (ncols < 2 || ncols > 3)
        throw std::invalid_argument("read_ascii_table: ncols must be 2 or 3");

    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open '" + path + "'");

    std::vector<std::array<double,3>> rows;
    std::string line;
    while (std::getline(in, line))
    {
        // trim leading whitespace
        auto it  = std::find_if_not(line.begin(), line.end(), ::isspace);
        if (it == line.end()) continue;           // blank line
        if (*it == comment_char) continue;        // comment

        std::istringstream ss(line);
        std::array<double,3> row{0.0,0.0,std::numeric_limits<double>::quiet_NaN()};

        if (ncols == 2)
        {
            if (!(ss >> row[0] >> row[1])) continue;
        } else {
            if (!(ss >> row[0] >> row[1] >> row[2])) continue;
        }
        rows.push_back(row);
    }
    if (rows.empty())
        throw std::runtime_error("File '" + path + "' contains no valid data");

    return rows;
}

// ----------------------------------------------------------------------------
//  Sort a vector of <λ, …> rows by λ ascending and move into Eigen vectors
// ----------------------------------------------------------------------------
template <int NC>
void to_eigen(const std::vector<std::array<double,NC>>& rows,
              Vector& l_out,
              Vector& f_out,
              Vector* s_out = nullptr)
{
    const std::size_t n = rows.size();
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](std::size_t i, std::size_t j)
              { return rows[i][0] < rows[j][0]; });

    l_out.resize(n);
    f_out.resize(n);
    if (s_out) s_out->resize(n);

    for (std::size_t k = 0; k < n; ++k)
    {
        const auto& r = rows[idx[k]];
        l_out[k] = r[0];
        f_out[k] = r[1];
        if (s_out) (*s_out)[k] = r[2];
    }
}

} // unnamed namespace
// ============================================================================
//  Public loader implementations
// ============================================================================

Spectrum load_ascii_2col(const std::string& path)
{
    // -------------------- 1. read file -------------------------------------------------
    const auto rows = read_ascii_table(path, /*ncols=*/2);

    // -------------------- 2. move & sort ----------------------------------------------
    Vector lambda, flux;
    to_eigen<3>(rows, lambda, flux);

    // -------------------- 3. estimate per-pixel noise ---------------------------------
    const Eigen::Index len = lambda.size();

    int npix_box = std::max(700,
                            static_cast<int>(std::round(len / 10.0)));
    npix_box     = std::min(npix_box,
                            static_cast<int>(std::round(len / 4.0)));

    SNRCurveResult snr;
    if (npix_box <= 700)
        snr = der_snr_curve(lambda, flux, npix_box);
    else
        snr = get_signal_to_noise_curve(lambda, flux, npix_box);

    // build extended table [λ₀ , λ_curve , λ_last]   similar for noise
    Vector x_old(snr.lambda.size() + 2);
    Vector y_old(snr.noise .size() + 2);

    x_old[0]                      = lambda[0];
    x_old[x_old.size() - 1]       = lambda[len - 1];
    y_old[0]                      = snr.noise[0];
    y_old[y_old.size() - 1]       = snr.noise[snr.noise.size() - 1];
    x_old.segment(1, snr.lambda.size()) = snr.lambda;
    y_old.segment(1, snr.noise .size()) = snr.noise;

    Vector sigma = linear_interpolate(lambda, x_old, y_old);


    // -------------------- 4. normalise by the median ----------------------------------
    const double med = median(flux);
    if (!std::isfinite(med) || std::abs(med) == 0.0)
        throw std::runtime_error("load_ascii_2col: invalid flux median");

    flux  .array() /= med;
    sigma .array() /= med;


    return Spectrum{lambda, flux, sigma};
}

// ----------------------------------------------------------------------------
Spectrum load_ascii_3col(const std::string& path)
{
    // -------------------- 1. read file -------------------------------------------------
    const auto rows = read_ascii_table(path, /*ncols=*/3);

    // -------------------- 2. move & sort ----------------------------------------------
    Vector lambda, flux, sigma;
    to_eigen<3>(rows, lambda, flux, &sigma);

    // -------------------- 3. normalise by the median ----------------------------------
    const double med = median(flux);
    if (!std::isfinite(med) || std::abs(med) == 0.0)
        throw std::runtime_error("load_ascii_3col: invalid flux median");

    flux .array() /= med;
    sigma.array() /= med;

    return Spectrum{lambda, flux, sigma};
}

// ============================================================================
//  Dispatcher / auto-detection  (unchanged stubs follow)
// ============================================================================

// keep previously generated stub macro for the remaining formats -------------
#define SPECFIT_MAKE_STUB(name)                                  \
Spectrum name(const std::string& path)                           \
{                                                                \
    throw std::runtime_error(#name ": loader not implemented");  \
}

SPECFIT_MAKE_STUB(load_sdss)
SPECFIT_MAKE_STUB(load_sdss_v)
SPECFIT_MAKE_STUB(load_lamost)
SPECFIT_MAKE_STUB(load_lamost_dr8)
SPECFIT_MAKE_STUB(load_feros)
SPECFIT_MAKE_STUB(load_feros_phase3)
SPECFIT_MAKE_STUB(load_uves)
SPECFIT_MAKE_STUB(load_xshooter)
SPECFIT_MAKE_STUB(load_xshooter_esoreflex)
SPECFIT_MAKE_STUB(load_fuse)
SPECFIT_MAKE_STUB(load_4most)
SPECFIT_MAKE_STUB(load_iraf)
SPECFIT_MAKE_STUB(load_muse)

#undef SPECFIT_MAKE_STUB

// ----------------------------------------------------------------------------
// Dispatcher / auto-detection -------------------------------------------------
// (same code as before – left unchanged)
// ----------------------------------------------------------------------------
static const std::unordered_map<std::string, SpectrumLoader> kLoaderMap = {
    {"ASCII_with_2_columns",  load_ascii_2col},
    {"ASCII_with_3_columns",  load_ascii_3col},
    {"SDSS",                  load_sdss},
    {"SDSSV",                 load_sdss_v},
    {"LAMOST",                load_lamost},
    {"LAMOST_DR8",            load_lamost_dr8},
    {"FEROS",                 load_feros},
    {"FEROS_phase3",          load_feros_phase3},
    {"UVES",                  load_uves},
    {"XSHOOTER",              load_xshooter},
    {"XSHOOTER_esoreflex",    load_xshooter_esoreflex},
    {"FUSE",                  load_fuse},
    {"4MOST",                 load_4most},
    {"IRAF",                  load_iraf},
    {"MUSE",                  load_muse}
};

Spectrum load_spectrum(const std::string& path, const std::string& format)
{
    if (format == "auto") {
        for (const auto& kv : kLoaderMap) {
            try { return kv.second(path); }
            catch (const std::exception&) { /* try next */ }
        }
        throw std::runtime_error("load_spectrum(auto): none of the "
                                 "registered readers could load '" + path + "'");
    }

    auto it = kLoaderMap.find(format);
    if (it == kLoaderMap.end())
        throw std::runtime_error("Unsupported spectrum format: " + format);

    return it->second(path);
}

} // namespace specfit