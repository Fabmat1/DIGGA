#pragma once
#include "Types.hpp"
#include <string>
#include <vector>

namespace specfit {

// Container for an (optionally rebinned) spectrum
struct Spectrum {
    Vector               lambda;       // Å
    Vector               flux;         // arbitrary units
    Vector               sigma;        // 1-σ uncertainties (same units as flux)

    /* 1 == use point, 0 == ignore point during the optimisation          *
     * The size is always identical to lambda.size()                      */
    std::vector<int>     ignoreflag;
};

/* unchanged */
Spectrum load_ascii(const std::string& path, bool three_col);

} // namespace specfit