
#pragma once

#include <vector>

#include "tile/lang/ops.h"

namespace vertexai {
namespace tile {
namespace lang {

// Remove any fractional polynomial multipliers (IE, any non-integers).
Contraction Defract(const Contraction& in, const std::vector<RangeConstraint>& order);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
