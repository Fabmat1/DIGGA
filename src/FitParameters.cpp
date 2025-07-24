#include "specfit/FitParameters.hpp"
#include <stdexcept>

namespace specfit {

void FitParameters::set(const std::string& name, double val, bool frozen)
{
    p_[name] = {val, frozen};
}

Parameter& FitParameters::operator[](const std::string& name)
{
    return p_[name];
}

const Parameter& FitParameters::at(const std::string& name) const
{
    auto it = p_.find(name);
    if (it == p_.end()) throw std::out_of_range(name);
    return it->second;
}

} // namespace specfit