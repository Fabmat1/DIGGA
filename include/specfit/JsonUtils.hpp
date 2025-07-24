#pragma once
#include <nlohmann/json.hpp>
#include <string>

namespace specfit {
nlohmann::json load_json(const std::string& path);
void expand_env(nlohmann::json& j);
} // namespace specfit