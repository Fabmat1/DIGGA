#pragma once
#include "Spectrum.hpp"
#include "Types.hpp"
#include <string>
#include <vector>

namespace specfit {

struct GridAxis {
    std::string name;
    Vector      values;
};

class ModelGrid {
public:
    // Resolve <base_path>/<rel_path>/grid.fits; first hit wins.
    ModelGrid(const std::vector<std::string>& base_paths,
              const std::string& rel_path);
    explicit ModelGrid(std::string abs_path);

    Spectrum load_spectrum(double teff,
                           double logg,
                           double z,
                           double he,
                           double xi,
                           double vsini,     // Added vsini parameter
                           double resOffset,
                           double resSlope) const;

    const std::vector<GridAxis>& axes() const { return axes_; }

    struct ParameterBounds {
        double teff_min = -std::numeric_limits<double>::max();
        double teff_max = std::numeric_limits<double>::max();
        double logg_min = -std::numeric_limits<double>::max();
        double logg_max = std::numeric_limits<double>::max();
        double z_min = -std::numeric_limits<double>::max();
        double z_max = std::numeric_limits<double>::max();
        double he_min = -std::numeric_limits<double>::max();
        double he_max = std::numeric_limits<double>::max();
        double xi_min = -std::numeric_limits<double>::max();
        double xi_max = std::numeric_limits<double>::max();
        
        // Convert to vectors for LM solver (order: teff, logg, z, he, xi)
        std::vector<double> get_lower_bounds() const {
            return {teff_min, logg_min, z_min, he_min, xi_min};
        }
        
        std::vector<double> get_upper_bounds() const {
            return {teff_max, logg_max, z_max, he_max, xi_max};
        }
        
        // Check if parameters are at boundary (within tolerance)
        std::vector<bool> at_lower_boundary(const Eigen::VectorXd& params, 
                                            double tol = 1e-6) const {
            std::vector<bool> at_boundary(5, false);
            if (params.size() >= 5) {
                at_boundary[0] = (params[0] - teff_min) < tol * std::abs(teff_min);
                at_boundary[1] = (params[1] - logg_min) < tol * std::abs(logg_min);
                at_boundary[2] = (params[2] - z_min) < tol * std::abs(z_min);
                at_boundary[3] = (params[3] - he_min) < tol * std::abs(he_min);
                at_boundary[4] = (params[4] - xi_min) < tol * std::abs(xi_min);
            }
            return at_boundary;
        }
        
        std::vector<bool> at_upper_boundary(const Eigen::VectorXd& params,
                                            double tol = 1e-6) const {
            std::vector<bool> at_boundary(5, false);
            if (params.size() >= 5) {
                at_boundary[0] = (teff_max - params[0]) < tol * std::abs(teff_max);
                at_boundary[1] = (logg_max - params[1]) < tol * std::abs(logg_max);
                at_boundary[2] = (z_max - params[2]) < tol * std::abs(z_max);
                at_boundary[3] = (he_max - params[3]) < tol * std::abs(he_max);
                at_boundary[4] = (xi_max - params[4]) < tol * std::abs(xi_max);
            }
            return at_boundary;
        }
    };
    
    ParameterBounds get_parameter_bounds() const;

private:
    std::string              base_;
    std::vector<GridAxis>    axes_;

    Spectrum read_fits(const std::string& path) const;
};

} // namespace specfit