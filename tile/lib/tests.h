#pragma once

#include <map>
#include <optional>
#include <string>

#include "plaidml2/edsl/edsl.h"

namespace vertexai::tile::lib {

using plaidml2::edsl::Program;

void RegisterTest(const std::string& name, std::function<Program()> factory);
std::optional<Program> CreateTest(const std::string& name);

}  // namespace vertexai::tile::lib
