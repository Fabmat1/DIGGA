#include "specfit/Spectrum.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace specfit {

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