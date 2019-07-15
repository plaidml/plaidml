// Copyright 2019, Intel Corporation

#include "tile/plaid_ir/ops.h"
#include "mlir/IR/OpImplementation.h"

namespace vertexai {
namespace tile {
namespace plaid_ir {

#define GET_OP_CLASSES
#include "tile/plaid_ir/ops.cpp.inc"

}  // namespace plaid_ir
}  // namespace tile
}  // namespace vertexai
