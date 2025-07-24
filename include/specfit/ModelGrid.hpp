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
                           double resOffset,
                           double resSlope) const;

    const std::vector<GridAxis>& axes() const { return axes_; }

private:
    std::string              base_;
    std::vector<GridAxis>    axes_;

    Spectrum read_fits(const std::string& path) const;
};

} // namespace specfit