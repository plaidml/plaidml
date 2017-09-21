#pragma once

#include <vector>

#include "tile/lang/ops.h"

namespace vertexai {
namespace tile {
namespace lang {

Contraction ReduceOutputPolynomials(const Contraction& op, const std::vector<RangeConstraint>& order);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
