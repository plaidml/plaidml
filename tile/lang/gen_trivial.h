#pragma once

#include <string>

#include "tile/base/shape.h"
#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace lang {

KernelInfo GenCopy(const TensorShape& shape, const std::string& oname, const std::string& iname,
                   const std::string& kname);
KernelInfo GenZero(const TensorShape& shape, const std::string& bname, const std::string& kname);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
