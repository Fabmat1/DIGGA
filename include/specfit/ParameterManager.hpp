// include/specfit/ParameterManager.hpp
#pragma once
#include "Types.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace specfit {

struct ParameterInfo {
    std::string name;
    double* ptr;
    bool frozen;
    double lower_bound = -std::numeric_limits<double>::infinity();
    double upper_bound = std::numeric_limits<double>::infinity();
};

class ParameterManager {
public:
    void add_parameter(const std::string& name, double* ptr, bool frozen, 
                      double lower = -std::numeric_limits<double>::infinity(),
                      double upper = std::numeric_limits<double>::infinity());
    
    std::vector<double*> get_free_parameters() const;
    std::vector<double*> get_all_parameters() const;
    bool is_frozen(const std::string& name) const;
    
    // For Ceres parameter blocks
    std::vector<double*> build_parameter_blocks() const;
    std::vector<int> get_block_sizes() const;
    
private:
    std::vector<ParameterInfo> params_;
    std::unordered_map<std::string, size_t> name_to_idx_;
};

} // namespace specfit