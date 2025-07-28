#include "specfit/SNRHelpers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace specfit {
namespace {

// ============================================================================
//  Small helpers (local linkage)
// ============================================================================
inline double
mean(const Eigen::VectorXd& v)
{
    return v.size() ? v.mean() : std::numeric_limits<double>::quiet_NaN();
}

// ----------------------------------------------------------------------------
double
standard_deviation(const Eigen::VectorXd& v, double mu)
{
    if (v.size() < 2) return 0.0;
    const double var =
        (v.array() - mu).square().sum() / static_cast<double>(v.size() - 1);
    return std::sqrt(var);
}

// ----------------------------------------------------------------------------
double
median(const Eigen::VectorXd& v)
{
    const Eigen::Index n = v.size();
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();

    std::vector<double> tmp(v.data(), v.data() + n);
    const auto mid = tmp.begin() + n / 2;
    std::nth_element(tmp.begin(), mid, tmp.end());
    if (n % 2 == 1)                 // odd
        return *mid;

    // even – need mean of two central entries
    const auto max_it = std::max_element(tmp.begin(), mid);
    return (*max_it + *mid) * 0.5;
}

// ----------------------------------------------------------------------------
//  Piece-wise linear interpolation: y(x_old) -> y_interp(x_new)
// ----------------------------------------------------------------------------
Eigen::VectorXd
linear_interpolate(const Eigen::VectorXd& x_new,
                   const Eigen::VectorXd& x_old,
                   const Eigen::VectorXd& y_old)
{
    if (x_old.size() != y_old.size() || x_old.size() < 2)
        throw std::invalid_argument("Invalid interpolation table.");

    Eigen::VectorXd y_new(x_new.size());

    for (Eigen::Index i = 0; i < x_new.size(); ++i)
    {
        const double x = x_new[i];

        // left or right of table: clamp
        if (x <= x_old[0]) { y_new[i] = y_old[0];  continue; }
        if (x >= x_old[x_old.size()-1])
        {
            y_new[i] = y_old[y_old.size()-1];
            continue;
        }

        // binary search for segment
        const auto it = std::upper_bound(x_old.data(),
                                         x_old.data() + x_old.size(), x);
        const Eigen::Index hi = static_cast<Eigen::Index>(it - x_old.data());
        const Eigen::Index lo = hi - 1;

        const double t =
            (x - x_old[lo]) / (x_old[hi] - x_old[lo]);   // 0 … 1
        y_new[i] = (1.0 - t) * y_old[lo] + t * y_old[hi];
    }
    return y_new;
}

} // unnamed namespace


// ============================================================================
//  get_signal_to_noise   (≈ original get_signal_to_noise.sl without fitting)
// ============================================================================
SNRResult
get_signal_to_noise(const Eigen::VectorXd& flux, int neighbor)
{
    const Eigen::Index n = flux.size();
    if (n < 2 * neighbor + 1)
        throw std::invalid_argument(
            "get_signal_to_noise: flux vector too short for chosen neighbour.");

    // ------------------------------------------------------------------
    // build ∆ᵢ = fᵢ – ½(fᵢ₋ₙ + fᵢ₊ₙ)   valid for i = n … N-1-n
    // ------------------------------------------------------------------
    const Eigen::Index m = n - 2 * neighbor;
    Eigen::VectorXd ndist(m);

    for (Eigen::Index i = 0; i < m; ++i)
    {
        const Eigen::Index idx = i + neighbor;
        ndist[i] = flux[idx]
                 - 0.5 * (flux[idx - neighbor] + flux[idx + neighbor]);
    }

    // check for constant flux
    if ((ndist.array() == 0.0).all())
    {
        return {0.0, std::numeric_limits<double>::infinity()};
    }

    // ------------------------------------------------------------------
    // 2-sigma clipping to remove outliers
    // ------------------------------------------------------------------
    {
        double mu  = mean(ndist);
        double sig = standard_deviation(ndist, mu);

        const auto mask =
            ( (ndist.array() >= (mu - 2.0*sig)) &&
              (ndist.array() <= (mu + 2.0*sig)) );

        const Eigen::VectorXd clipped =
            ndist(mask.matrix().cast<int>().template cast<bool>());

        ndist = clipped;     // overwrite with clipped subset
    }

    // final statistics
    const double mu  = mean(ndist);
    const double sig = standard_deviation(ndist, mu);    // σ(∆ᵢ)

    // conversion: σ(flux) = σ(∆ᵢ) / sqrt(3/2)
//    constexpr double conversion = std::sqrt(1.5);
    constexpr double conversion = 1.224744871391589;
    const double noise = sig / conversion;

    const double signal = median(flux);
    const double snr    = noise > 0.0
                        ? signal / noise
                        : std::numeric_limits<double>::infinity();

    return {noise, snr};
}


// ============================================================================
//  get_signal_to_noise_curve
// ============================================================================
SNRCurveResult
get_signal_to_noise_curve(const Eigen::VectorXd& lambda,
                          const Eigen::VectorXd& flux,
                          int  data_points,
                          int  neighbor)
{
    const Eigen::Index len = lambda.size();
    if (flux.size() != len)
        throw std::invalid_argument("lambda / flux length mismatch.");

    if (data_points < 1) data_points = 1;
    if (data_points > static_cast<int>(len))
        data_points = static_cast<int>(len);

    // convert “number of data points” to “last index offset” (S-Lang did -1)
    --data_points;

    std::vector<double> v_lambda, v_noise, v_snr;
    int index = 0;

    while (index + data_points <= len - 1)
    {
        const int lo = index;
        const int hi = index + data_points;

        const Eigen::VectorXd slice = flux.segment(lo, hi - lo + 1);
        const SNRResult r = get_signal_to_noise(slice, neighbor);

        v_lambda.push_back(lambda[lo + data_points / 2]);
        v_noise .push_back(r.noise);
        v_snr   .push_back(r.snr);

        index += data_points / 2;       // 50 % overlap
    }

    // trailing window
    {
        const int lo = len - 1 - data_points;
        const int hi = len - 1;
        const Eigen::VectorXd slice = flux.segment(lo, hi - lo + 1);
        const SNRResult r = get_signal_to_noise(slice, neighbor);

        v_lambda.push_back(lambda[lo + data_points / 2]);
        v_noise .push_back(r.noise);
        v_snr   .push_back(r.snr);
    }

    const Eigen::Index n = static_cast<Eigen::Index>(v_lambda.size());
    SNRCurveResult out;
    out.lambda.resize(n);
    out.noise .resize(n);
    out.snr   .resize(n);

    for (Eigen::Index i = 0; i < n; ++i)
    {
        out.lambda[i] = v_lambda[i];
        out.noise [i] = v_noise [i];
        out.snr   [i] = v_snr  [i];
    }
    return out;
}


// ============================================================================
//  der_snr           (DER_SNR algorithm, order 3)
// ============================================================================
SNRResult
der_snr(const Eigen::VectorXd& flux)
{
    const Eigen::Index n = flux.size();
    constexpr int order   = 3;
    constexpr int npixmin = 2*order;        // needs at least 6 pixels

    if (n < npixmin)
        throw std::invalid_argument("der_snr: not enough data points.");

    const double signal = median(flux);

    // DER_SNR constants (order 3 -> f3)
    constexpr double f3 = 0.6052697319;

    // 2·fᵢ – fᵢ₋₂ – fᵢ₊₂     valid for i = 2 … n-3
    const Eigen::Index m = n - 2*order;
    Eigen::VectorXd diffs(m);

    for (Eigen::Index i = 0; i < m; ++i)
    {
        const Eigen::Index idx = i + order;
        diffs[i] = 2.0 * flux[idx] - flux[idx - order] - flux[idx + order];
    }

    const double noise = f3 * median(diffs.cwiseAbs());
    const double snr   = noise > 0.0
                       ? signal / noise
                       : std::numeric_limits<double>::infinity();

    return {noise, snr};
}


// ============================================================================
//  der_snr_curve
// ============================================================================
SNRCurveResult
der_snr_curve(const Eigen::VectorXd& lambda,
              const Eigen::VectorXd& flux,
              int  data_points)
{
    const Eigen::Index len = lambda.size();
    if (flux.size() != len)
        throw std::invalid_argument("lambda / flux length mismatch.");

    if (data_points < 1) data_points = 1;
    if (data_points > static_cast<int>(len))
        data_points = static_cast<int>(len);

    --data_points;       // index offset like above

    std::vector<double> v_lambda, v_noise, v_snr;
    int index = 0;

    while (index + data_points <= len - 1)
    {
        const int lo = index;
        const int hi = index + data_points;

        const Eigen::VectorXd slice = flux.segment(lo, hi - lo + 1);
        const SNRResult r = der_snr(slice);

        v_lambda.push_back(lambda[lo + data_points / 2]);
        v_noise .push_back(r.noise);
        v_snr   .push_back(r.snr);

        index += data_points / 2;
    }

    // trailing window
    {
        const int lo = len - 1 - data_points;
        const int hi = len - 1;
        const Eigen::VectorXd slice = flux.segment(lo, hi - lo + 1);
        const SNRResult r = der_snr(slice);

        v_lambda.push_back(lambda[lo + data_points / 2]);
        v_noise .push_back(r.noise);
        v_snr   .push_back(r.snr);
    }

    const Eigen::Index n = static_cast<Eigen::Index>(v_lambda.size());
    SNRCurveResult out;
    out.lambda.resize(n);
    out.noise .resize(n);
    out.snr   .resize(n);

    for (Eigen::Index i = 0; i < n; ++i)
    {
        out.lambda[i] = v_lambda[i];
        out.noise [i] = v_noise [i];
        out.snr   [i] = v_snr  [i];
    }
    return out;
}


// ============================================================================
//  snr_curve   (convenience wrapper)
// ============================================================================
SNRCurveResult
snr_curve(const Eigen::VectorXd& lambda,
          const Eigen::VectorXd& flux,
          const std::string&     method,
          int   data_points,
          bool  interpolate_back,
          int   neighbor)
{
    if (method != "der_snr" && method != "gauss")
        throw std::invalid_argument("snr_curve: method must be 'der_snr' or 'gauss'.");

    SNRCurveResult base;

    if (method == "der_snr")
        base = der_snr_curve(lambda, flux, data_points);
    else
        base = get_signal_to_noise_curve(lambda, flux, data_points, neighbor);

    if (!interpolate_back)
        return base;

    // ------------------------------------------------------------------
    //  Interpolate noise back to the native wavelength grid
    // ------------------------------------------------------------------
    const int iedge = std::max(
            std::min(data_points, static_cast<int>(lambda.size()) - 1),
            static_cast<int>(lambda.size() / 20));

    // build extended table (mimics original edge handling)
    Eigen::VectorXd x_old(base.lambda.size() + 2);
    Eigen::VectorXd y_old(base.noise.size()  + 2);

    x_old[0]                   = lambda[0];
    x_old[x_old.size() - 1]    = lambda[lambda.size() - 1];

    y_old[0]                   = (method == "der_snr")
                                 ? der_snr( flux.head(iedge+1) ).noise
                                 : get_signal_to_noise( flux.head(iedge+1),
                                                        neighbor ).noise;

    y_old[y_old.size() - 1]    = (method == "der_snr")
                                 ? der_snr( flux.tail(iedge+1) ).noise
                                 : get_signal_to_noise( flux.tail(iedge+1),
                                                        neighbor ).noise;

    // insert core points
    x_old.segment(1, base.lambda.size()) = base.lambda;
    y_old.segment(1, base.noise .size()) = base.noise;

    const Eigen::VectorXd err = linear_interpolate(lambda, x_old, y_old);
    if (err.size() != lambda.size())
        throw std::runtime_error("Interpolation failed.");

    SNRCurveResult out;
    out.lambda = lambda;
    out.noise  = err;
    out.snr    = flux.cwiseQuotient(err);

    return out;
}

} // namespace specfit
