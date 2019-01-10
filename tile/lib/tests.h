#pragma once

#include <map>
#include <string>

#include "tile/lang/compose.h"

namespace vertexai {
namespace tile {
namespace lib {

std::map<std::string, lang::RunInfo> InternalTests();

}  // namespace lib
}  // namespace tile
}  // namespace vertexai
