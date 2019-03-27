#include "tile/ocl_exec/intrinsic.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

std::vector<lang::IntrinsicSpec> ocl_intrinsics = {
    {"vloadn", EmitVloadn},
    {"vstoren", EmitVstoren},
};

sem::ExprPtr EmitVloadn(const stripe::Intrinsic& in) { return nullptr; }

sem::ExprPtr EmitVstoren(const stripe::Intrinsic& in) { return nullptr; }

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
