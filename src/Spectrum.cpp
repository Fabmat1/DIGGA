#include "specfit/Spectrum.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace specfit {

// Helper function to compute median
static Real median(Vector vec) {
    auto n = vec.size();
    if (n == 0) return 0.0;
    
    std::sort(vec.data(), vec.data() + n);
    if (n % 2 == 0) {
        return 0.5 * (vec[n/2 - 1] + vec[n/2]);
    } else {
        return vec[n/2];
    }
}

// Helper function for linear interpolation
static Vector interpolate(const Vector& x_new, const Vector& x_old, const Vector& y_old) {
    Vector y_new(x_new.size());
    
    for (int i = 0; i < x_new.size(); ++i) {
        Real x = x_new[i];
        
        // Find bracketing indices
        int j = 0;
        while (j < x_old.size() - 1 && x_old[j+1] < x) {
            ++j;
        }
        
        if (j == x_old.size() - 1) {
            y_new[i] = y_old[j];
        } else if (j == 0 && x < x_old[0]) {
            y_new[i] = y_old[0];
        } else {
            // Linear interpolation
            Real t = (x - x_old[j]) / (x_old[j+1] - x_old[j]);
            y_new[i] = y_old[j] * (1 - t) + y_old[j+1] * t;
        }
    }
    
    return y_new;
}

Vector Spectrum::compute_delta_distribution(const Vector& flux_vec, int neighbor) const {
    int n = flux_vec.size();
    Vector delta(n);
    
    for (int i = neighbor; i < n - neighbor; ++i) {
        delta[i] = flux_vec[i] - 0.5 * (flux_vec[i - neighbor] + flux_vec[i + neighbor]);
    }
    
    // Handle edges
    for (int i = 0; i < neighbor; ++i) {
        delta[i] = delta[neighbor];
    }
    for (int i = n - neighbor; i < n; ++i) {
        delta[i] = delta[n - neighbor - 1];
    }
    
    return delta;
}

SNRResult Spectrum::estimate_snr_gaussian_impl(const Vector& flux_vec, int neighbor) const {
    SNRResult result;
    
    if (flux_vec.size() < 2 * neighbor + 1) {
        result.noise = 0.0;
        result.snr = std::numeric_limits<Real>::infinity();
        return result;
    }
    
    // Compute delta distribution
    Vector delta = compute_delta_distribution(flux_vec, neighbor);
    
    // Remove outliers (keep values within 2 sigma)
    Real mean = delta.mean();
    Real stddev = std::sqrt((delta.array() - mean).square().mean());
    
    std::vector<Real> filtered;
    for (int i = 0; i < delta.size(); ++i) {
        if (std::abs(delta[i] - mean) <= 2 * stddev) {
            filtered.push_back(delta[i]);
        }
    }
    
    if (filtered.empty()) {
        result.noise = 0.0;
        result.snr = std::numeric_limits<Real>::infinity();
        return result;
    }
    
    // Recompute statistics on filtered data
    Vector filtered_vec = Eigen::Map<Vector>(filtered.data(), filtered.size());
    stddev = std::sqrt((filtered_vec.array() - filtered_vec.mean()).square().mean());
    
    // Conversion factor from delta distribution to actual noise
    const Real conversion_factor = std::sqrt(1.5);
    
    result.noise = stddev / conversion_factor;
    result.snr = median(flux_vec) / result.noise;
    
    return result;
}

SNRResult Spectrum::estimate_snr_der_impl(const Vector& flux_vec, int order) const {
    SNRResult result;
    
    const int npixmin = 6;
    int n = flux_vec.size();
    
    if (n < npixmin) {
        throw std::runtime_error("Not enough data points for DER_SNR estimation");
    }
    
    // 3rd order DER_SNR
    if (order == 3) {
        const Real f3 = 0.6052697319;
        
        Vector diff(n - 4);
        for (int i = 2; i < n - 2; ++i) {
            diff[i - 2] = std::abs(2.0 * flux_vec[i] - flux_vec[i - 2] - flux_vec[i + 2]);
        }
        
        result.noise = f3 * median(diff);
        result.snr = median(flux_vec) / result.noise;
    } else {
        throw std::runtime_error("Only 3rd order DER_SNR is implemented");
    }
    
    return result;
}

SNRResult Spectrum::estimate_snr_gaussian(int neighbor) const {
    return estimate_snr_gaussian_impl(flux, neighbor);
}

SNRResult Spectrum::estimate_snr_der(int order) const {
    return estimate_snr_der_impl(flux, order);
}

SNRCurve Spectrum::estimate_snr_curve(const std::string& method, 
                                      int window_size,
                                      int neighbor) const {
    SNRCurve curve;
    
    int n = lambda.size();
    if (window_size > n) {
        window_size = n;
    }
    
    std::vector<Real> lambda_vec, noise_vec, snr_vec;
    
    // Slide window across spectrum
    int step = window_size / 2;
    for (int start = 0; start + window_size <= n; start += step) {
        int end = start + window_size;
        Vector window_flux = flux.segment(start, window_size);
        
        SNRResult result;
        if (method == "gauss" || method == "gaussian") {
            result = estimate_snr_gaussian_impl(window_flux, neighbor);
        } else if (method == "der_snr") {
            result = estimate_snr_der_impl(window_flux, 3);
        } else {
            throw std::runtime_error("Unknown SNR estimation method: " + method);
        }
        
        lambda_vec.push_back(lambda[start + window_size / 2]);
        noise_vec.push_back(result.noise);
        snr_vec.push_back(result.snr);
    }
    
    // Handle last window if not already covered
    if (lambda_vec.empty() || lambda_vec.back() < lambda[n - window_size / 2]) {
        int start = n - window_size;
        Vector window_flux = flux.segment(start, window_size);
        
        SNRResult result;
        if (method == "gauss" || method == "gaussian") {
            result = estimate_snr_gaussian_impl(window_flux, neighbor);
        } else {
            result = estimate_snr_der_impl(window_flux, 3);
        }
        
        lambda_vec.push_back(lambda[n - window_size / 2]);
        noise_vec.push_back(result.noise);
        snr_vec.push_back(result.snr);
    }
    
    curve.lambda = Eigen::Map<Vector>(lambda_vec.data(), lambda_vec.size());
    curve.noise = Eigen::Map<Vector>(noise_vec.data(), noise_vec.size());
    curve.snr = Eigen::Map<Vector>(snr_vec.data(), snr_vec.size());
    
    return curve;
}

void Spectrum::compute_snr_errors(Vector& errors_out,
                                  Vector& snr_out,
                                  const std::string& method,
                                  int window_size) const {
    // Get SNR curve
    SNRCurve curve = estimate_snr_curve(method, window_size);
    
    // Extend endpoints for interpolation
    int n = lambda.size();
    int edge_size = std::min(window_size, n - 1);
    
    Vector lambda_extended(curve.lambda.size() + 2);
    Vector noise_extended(curve.noise.size() + 2);
    
    // Add edge estimates
    lambda_extended[0] = lambda[0];
    lambda_extended[lambda_extended.size() - 1] = lambda[n - 1];
    lambda_extended.segment(1, curve.lambda.size()) = curve.lambda;
    
    // Compute edge noise estimates
    Vector edge_flux_start = flux.head(edge_size);
    Vector edge_flux_end = flux.tail(edge_size);
    
    SNRResult edge_start, edge_end;
    if (method == "gauss" || method == "gaussian") {
        edge_start = estimate_snr_gaussian_impl(edge_flux_start, 2);
        edge_end = estimate_snr_gaussian_impl(edge_flux_end, 2);
    } else {
        edge_start = estimate_snr_der_impl(edge_flux_start, 3);
        edge_end = estimate_snr_der_impl(edge_flux_end, 3);
    }
    
    noise_extended[0] = edge_start.noise;
    noise_extended[noise_extended.size() - 1] = edge_end.noise;
    noise_extended.segment(1, curve.noise.size()) = curve.noise;
    
    // Interpolate back to original wavelength grid
    errors_out = interpolate(lambda, lambda_extended, noise_extended);
    
    // Compute SNR at each point
    snr_out = flux.array() / errors_out.array();
}

// [Previous load_ascii implementation remains unchanged]
static Spectrum load_ascii_impl(const std::string& path, int cols)
{
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);

    std::vector<Real> l, fl, sig;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        Real a, b, c = 0.0;
        if (cols == 2) {
            iss >> a >> b;
            c = b > 0 ? std::sqrt(b) : 1.0;
        } else {
            iss >> a >> b >> c;
        }
        l .push_back(a);
        fl.push_back(b);
        sig.push_back((c > 0.0 && std::isfinite(c)) ? c : 1.0); // σ safeguard
    }
    Spectrum s;
    s.lambda = Eigen::Map<Vector>(l.data(), l.size());
    s.flux   = Eigen::Map<Vector>(fl.data(), fl.size());
    s.sigma  = Eigen::Map<Vector>(sig.data(), sig.size());
    return s;
}

Spectrum load_ascii(const std::string& path, bool three_col)
{
    Spectrum sp = load_ascii_impl(path, three_col ? 3 : 2);
    sp.ignoreflag.assign(sp.lambda.size(), 1);
    return sp;
}

} // namespace specfit