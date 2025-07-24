#ifndef SPECFIT_SNR_HELPERS_HPP
#define SPECFIT_SNR_HELPERS_HPP
// -----------------------------------------------------------------------------
//  Small toolbox to estimate signal–to–noise ratios for astronomical spectra.
//  The implementation is a modernised C++ translation of the original S-Lang
//  routines written for ISIS.  Arrays are represented by Eigen::VectorXd.
// -----------------------------------------------------------------------------
#include <Eigen/Dense>
#include <string>

namespace specfit {

// -----------------------------------------------------------------------------
//  Small return structures
// -----------------------------------------------------------------------------
struct SNRResult
{
    double noise {0.0};                // 1σ noise level
    double snr   {0.0};                // signal / noise
};

struct SNRCurveResult
{
    Eigen::VectorXd lambda;            // central wavelength of each window
    Eigen::VectorXd noise;             // local 1σ noise
    Eigen::VectorXd snr;               // local signal / noise
};

// -----------------------------------------------------------------------------
//  Function prototypes
// -----------------------------------------------------------------------------
SNRResult get_signal_to_noise(
        const Eigen::VectorXd& flux,
        int  neighbor = 2);                            // default from S-Lang

SNRCurveResult get_signal_to_noise_curve(
        const Eigen::VectorXd& lambda,
        const Eigen::VectorXd& flux,
        int  data_points = 3000,                       // default from S-Lang
        int  neighbor     = 2);

SNRResult der_snr(const Eigen::VectorXd& flux);

SNRCurveResult der_snr_curve(
        const Eigen::VectorXd& lambda,
        const Eigen::VectorXd& flux,
        int  data_points = 300);

SNRCurveResult snr_curve(
        const Eigen::VectorXd& lambda,
        const Eigen::VectorXd& flux,
        const std::string&    method          = "der_snr", // "der_snr" | "gauss"
        int   data_points     = 300,
        bool  interpolate_back = true,
        int   neighbor         = 2);

} // namespace specfit
#endif // SPECFIT_SNR_HELPERS_HPP