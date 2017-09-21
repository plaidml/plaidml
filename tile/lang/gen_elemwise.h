#pragma once

#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <vector>

#include "tile/lang/flat.h"
#include "tile/lang/generate.h"
#include "tile/lang/type.h"
#include "tile/lang/usedef.h"

namespace vertexai {
namespace tile {
namespace lang {

KernelInfo GenZero(const TensorShape& shape, const std::string& bname, const std::string& kname);
KernelInfo GenFunction(const Program& prog, const ShapeMap& outputs, const ShapeMap& types, const Bindings& vars,
                       const std::set<size_t>& comps, const std::string& kname, const UseDef& ud,
                       const HardwareSettings& settings);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
