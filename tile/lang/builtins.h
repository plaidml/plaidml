#pragma once

#include <map>
#include <memory>
#include <string>

#include "tile/lang/compose.h"

namespace vertexai {
namespace tile {
namespace lang {

extern std::map<std::string, Program> InlineDefines;
extern std::map<std::string, std::shared_ptr<BoundFunction>> DerivDefines;

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
