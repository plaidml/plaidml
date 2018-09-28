#pragma once

#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "tile/base/shape.h"
#include "tile/lang/flat.h"
#include "tile/lang/ops.h"

namespace vertexai {
namespace tile {
namespace lang {

FlatContraction Compile(const Contraction& c, const std::vector<TensorShape>& shapes,
                        std::vector<Polynomial<Rational>>* out_poly = nullptr);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
