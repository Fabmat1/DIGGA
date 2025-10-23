#pragma once
#include "SimpleLM.hpp"
#include "ModelGrid.hpp"
#include <iostream>
#include <iomanip>

namespace specfit {

class BoundedLMSolver {
public:
    // Enhanced solver summary with boundary information
    struct BoundedSolverSummary : public LMSolverSummary {
        std::vector<bool> at_lower_bound;
        std::vector<bool> at_upper_bound;
        std::vector<double> lower_uncertainties;  // Asymmetric uncertainties
        std::vector<double> upper_uncertainties;
        
        void print_summary() const {
            std::cout << "\n=== Fit Summary ===" << std::endl;
            std::cout << "Converged: " << (converged ? "Yes" : "No") << std::endl;
            std::cout << "Iterations: " << iterations << std::endl;
            std::cout << "Initial χ²: " << initial_chi2 << std::endl;
            std::cout << "Final χ²: " << final_chi2 << std::endl;
            std::cout << "\nParameter uncertainties:" << std::endl;
            
            const char* param_names[] = {"Teff", "log g", "[M/H]", "He", "ξ"};
            for (size_t i = 0; i < param_uncertainties.size() && i < 5; ++i) {
                std::cout << std::setw(8) << param_names[i] << ": ";
                
                if (at_lower_bound[i]) {
                    std::cout << "± " << upper_uncertainties[i] 
                             << " (at lower bound)";
                } else if (at_upper_bound[i]) {
                    std::cout << "± " << lower_uncertainties[i] 
                             << " (at upper bound)";
                } else {
                    std::cout << "± " << param_uncertainties[i];
                }
                std::cout << std::endl;
            }
        }
    };
    
    // Main fitting function with automatic boundary handling
    template<typename Functor>
    static BoundedSolverSummary fit_with_grid_bounds(
        Functor&& func,
        Eigen::VectorXd& params,
        const ModelGrid& grid,
        const std::vector<bool>& free_mask = {},
        const LMSolverOptions& options = {},
        LMWorkspace* workspace = nullptr)
    {
        // Get bounds from the grid
        auto grid_bounds = grid.get_parameter_bounds();
        auto lower = grid_bounds.get_lower_bounds();
        auto upper = grid_bounds.get_upper_bounds();
        
        // Add small margin to avoid numerical issues at exact boundaries
        const double margin = 1e-10;
        for (size_t i = 0; i < lower.size(); ++i) {
            lower[i] += margin * std::abs(lower[i]);
            upper[i] -= margin * std::abs(upper[i]);
        }
        
        // Ensure initial parameters are within bounds
        for (size_t i = 0; i < params.size() && i < lower.size(); ++i) {
            params[i] = std::clamp(params[i], lower[i], upper[i]);
        }
        
        // Run the standard LM solver with bounds
        auto base_summary = levenberg_marquardt(
            std::forward<Functor>(func),
            params,
            free_mask,
            lower,
            upper,
            options,
            workspace
        );
        
        // Convert to bounded summary
        BoundedSolverSummary summary;
        static_cast<LMSolverSummary&>(summary) = base_summary;
        
        // Check which parameters are at boundaries
        summary.at_lower_bound = grid_bounds.at_lower_boundary(params);
        summary.at_upper_bound = grid_bounds.at_upper_boundary(params);
        
        // Calculate asymmetric uncertainties for boundary parameters
        summary.lower_uncertainties = summary.param_uncertainties;
        summary.upper_uncertainties = summary.param_uncertainties;
        
        // Adjust uncertainties for parameters at boundaries
        for (size_t i = 0; i < params.size() && i < 5; ++i) {
            if (free_mask.size() > i && !free_mask[i]) continue;
            
            if (summary.at_lower_bound[i]) {
                // At lower bound: can only go up
                summary.lower_uncertainties[i] = 0.0;
                // Estimate upper uncertainty using one-sided derivative
                summary.upper_uncertainties[i] = estimate_one_sided_uncertainty(
                    func, params, i, true, summary.final_chi2
                );
            }
            else if (summary.at_upper_bound[i]) {
                // At upper bound: can only go down
                summary.upper_uncertainties[i] = 0.0;
                // Estimate lower uncertainty using one-sided derivative
                summary.lower_uncertainties[i] = estimate_one_sided_uncertainty(
                    func, params, i, false, summary.final_chi2
                );
            }
        }
        
        return summary;
    }
    
    private:
    // Estimate uncertainty using one-sided numerical derivative
    template<typename Functor>
    static double estimate_one_sided_uncertainty(
        Functor& func,
        const Eigen::VectorXd& params,
        int param_index,
        bool forward,  // true = forward difference, false = backward
        double chi2_min)
    {
        const double delta_chi2 = 1.0;  // 1-sigma corresponds to Δχ² = 1
        
        // Try different step sizes to find where Δχ² ≈ 1
        Eigen::VectorXd test_params = params;
        double h = 1e-3 * std::abs(params[param_index]);
        if (h == 0.0) h = 1e-3;
        
        // Binary search for the parameter value where Δχ² = 1
        double h_min = 0;
        double h_max = 10.0 * h;
        
        for (int iter = 0; iter < 20; ++iter) {
            test_params[param_index] = params[param_index] + (forward ? h : -h);
            
            Eigen::VectorXd residuals;
            func(test_params, &residuals, nullptr);
            double chi2 = residuals.squaredNorm();
            double delta = chi2 - chi2_min;
            
            if (std::abs(delta - delta_chi2) < 0.1) {
                // Close enough to Δχ² = 1
                return std::abs(h);
            }
            
            if (delta < delta_chi2) {
                h_min = std::abs(h);
                h = (h_min + h_max) / 2.0;
            } else {
                h_max = std::abs(h);
                h = (h_min + h_max) / 2.0;
            }
            
            if (h_max - h_min < 1e-6 * std::abs(params[param_index])) {
                break;
            }
        }
        
        return std::abs(h);
    }
};
} // namespace specfit