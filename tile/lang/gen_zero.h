#pragma once

#include <string>

#include "tile/lang/generate.h"
#include "tile/lang/shape.h"

namespace vertexai {
namespace tile {
namespace lang {

KernelInfo GenZero(const TensorShape& shape, const std::string& bname, const std::string& kname);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
