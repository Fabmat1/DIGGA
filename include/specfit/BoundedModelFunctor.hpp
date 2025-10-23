// BoundedModelFunctor.hpp

#pragma once
#include <Eigen/Core>
#include "ModelGrid.hpp"

namespace specfit {

class BoundedModelFunctor {
private:
    const ModelGrid& grid;
    const Vector& observed_flux;
    const Vector& observed_sigma;
    ModelGrid::ParameterBounds bounds;
    
    // Additional parameters that might be fixed or free
    double vsini_value;
    double res_offset_value;
    double res_slope_value;
    
    // Indices for parameters in the vector
    enum ParamIndex {
        TEFF = 0,
        LOGG = 1,
        Z = 2,
        HE = 3,
        XI = 4,
        VSINI = 5,
        RES_OFFSET = 6,
        RES_SLOPE = 7
    };
    
public:
    BoundedModelFunctor(const ModelGrid& g,
                        const Vector& obs_flux,
                        const Vector& obs_sigma,
                        double vsini = 0.0,
                        double res_offset = 0.0,
                        double res_slope = 0.0)
        : grid(g)
        , observed_flux(obs_flux)
        , observed_sigma(obs_sigma)
        , vsini_value(vsini)
        , res_offset_value(res_offset)
        , res_slope_value(res_slope)
    {
        bounds = grid.get_parameter_bounds();
    }
    
    void operator()(const Eigen::VectorXd& params,
                    Eigen::VectorXd* residuals,
                    Eigen::MatrixXd* jacobian) const
    {
        // Extract parameters (handle variable number of parameters)
        double teff = params[TEFF];
        double logg = params[LOGG];
        double z = (params.size() > Z) ? params[Z] : 0.0;
        double he = (params.size() > HE) ? params[HE] : 0.0;
        double xi = (params.size() > XI) ? params[XI] : 1.0;
        
        // Handle optional fitting of vsini and resolution parameters
        double vsini = (params.size() > VSINI) ? params[VSINI] : vsini_value;
        double res_offset = (params.size() > RES_OFFSET) ? params[RES_OFFSET] : res_offset_value;
        double res_slope = (params.size() > RES_SLOPE) ? params[RES_SLOPE] : res_slope_value;
        
        // Soft clamping with smooth transition near boundaries
        auto soft_clamp = [](double x, double min, double max) -> std::pair<double, double> {
            const double k = 100.0;  // Steepness of transition
            const double margin = (max - min) * 0.001;  // 0.1% margin
            
            if (x < min + margin) {
                double t = std::exp(-k * (x - min) / (max - min));
                double clamped = min + margin * std::exp(-t);
                double gradient = k * margin * t * std::exp(-t) / (max - min);
                return {clamped, gradient};
            } else if (x > max - margin) {
                double t = std::exp(k * (x - max) / (max - min));
                double clamped = max - margin * std::exp(-t);
                double gradient = -k * margin * t * std::exp(-t) / (max - min);
                return {clamped, gradient};
            }
            return {x, 1.0};  // No clamping needed, gradient = 1
        };
        
        // Apply soft clamping
        auto [teff_clamped, teff_grad] = soft_clamp(teff, bounds.teff_min, bounds.teff_max);
        auto [logg_clamped, logg_grad] = soft_clamp(logg, bounds.logg_min, bounds.logg_max);
        auto [z_clamped, z_grad] = soft_clamp(z, bounds.z_min, bounds.z_max);
        auto [he_clamped, he_grad] = soft_clamp(he, bounds.he_min, bounds.he_max);
        auto [xi_clamped, xi_grad] = soft_clamp(xi, bounds.xi_min, bounds.xi_max);
        
        // Load spectrum with clamped parameters
        Spectrum model = grid.load_spectrum(
            teff_clamped, logg_clamped, z_clamped, 
            he_clamped, xi_clamped, vsini, res_offset, res_slope
        );
        
        // Calculate residuals
        int n_data = observed_flux.size();
        if (residuals) {
            residuals->resize(n_data);
            for (int i = 0; i < n_data; ++i) {
                (*residuals)[i] = (observed_flux[i] - model.flux[i]) / observed_sigma[i];
            }
        }
        
        // Calculate Jacobian using finite differences with gradient correction
        if (jacobian) {
            jacobian->resize(n_data, params.size());
            
            const double eps = 1e-7;
            std::vector<double> grad_corrections = {
                teff_grad, logg_grad, z_grad, he_grad, xi_grad, 1.0, 1.0, 1.0
            };
            
            for (int j = 0; j < params.size(); ++j) {
                Eigen::VectorXd params_plus = params;
                Eigen::VectorXd params_minus = params;
                
                double h = eps * std::max(1.0, std::abs(params[j]));
                
                // Check if we can take centered difference
                bool use_forward = false;
                bool use_backward = false;
                
                // Determine which difference to use based on boundaries
                switch(j) {
                    case TEFF:
                        if (params[j] + h > bounds.teff_max) use_backward = true;
                        if (params[j] - h < bounds.teff_min) use_forward = true;
                        break;
                    case LOGG:
                        if (params[j] + h > bounds.logg_max) use_backward = true;
                        if (params[j] - h < bounds.logg_min) use_forward = true;
                        break;
                    case Z:
                        if (params[j] + h > bounds.z_max) use_backward = true;
                        if (params[j] - h < bounds.z_min) use_forward = true;
                        break;
                    case HE:
                        if (params[j] + h > bounds.he_max) use_backward = true;
                        if (params[j] - h < bounds.he_min) use_forward = true;
                        break;
                    case XI:
                        if (params[j] + h > bounds.xi_max) use_backward = true;
                        if (params[j] - h < bounds.xi_min) use_forward = true;
                        break;
                }
                
                Eigen::VectorXd r_plus, r_minus;
                
                if (use_forward) {
                    params_plus[j] += h;
                    (*this)(params_plus, &r_plus, nullptr);
                    jacobian->col(j) = (r_plus - *residuals) / h * grad_corrections[j];
                } else if (use_backward) {
                    params_minus[j] -= h;
                    (*this)(params_minus, &r_minus, nullptr);
                    jacobian->col(j) = (*residuals - r_minus) / h * grad_corrections[j];
                } else {
                    // Centered difference
                    params_plus[j] += h;
                    params_minus[j] -= h;
                    (*this)(params_plus, &r_plus, nullptr);
                    (*this)(params_minus, &r_minus, nullptr);
                    jacobian->col(j) = (r_plus - r_minus) / (2.0 * h) * grad_corrections[j];
                }
            }
        }
    }
};

} // namespace specfit