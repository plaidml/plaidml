#pragma once

#include <map>
#include <string>

#include <boost/optional.hpp>

#include "tile/lang/compose.h"

namespace vertexai {
namespace tile {
namespace lib {

void RegisterTest(const std::string& name, std::function<lang::RunInfo()> factory);
boost::optional<lang::RunInfo> CreateTest(const std::string& name);

}  // namespace lib
}  // namespace tile
}  // namespace vertexai
