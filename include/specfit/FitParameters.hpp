#pragma once
#include <unordered_map>
#include <string>

namespace specfit {

struct Parameter {
    double value;
    bool   frozen;
};

class FitParameters {
public:
    void set(const std::string& name, double val, bool frozen);
    Parameter& operator[](const std::string& name);
    const Parameter& at(const std::string& name) const;

private:
    std::unordered_map<std::string, Parameter> p_;
};

} // namespace specfit