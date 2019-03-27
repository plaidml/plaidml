#pragma once

#include <string>
#include <vector>

#include "tile/lang/intrinsic.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

sem::ExprPtr EmitVloadn(const stripe::Intrinsic& in);
sem::ExprPtr EmitVstoren(const stripe::Intrinsic& in);

extern std::vector<lang::IntrinsicSpec> ocl_intrinsics;

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
