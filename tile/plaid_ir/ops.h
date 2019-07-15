// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "tile/plaid_ir/types.h"

namespace vertexai {
namespace tile {
namespace plaid_ir {

using mlir::ArrayRef;
using mlir::Builder;
using mlir::LogicalResult;
using mlir::NamedAttribute;
using mlir::Op;
using mlir::Operation;
using mlir::OperationState;
using mlir::StringRef;
using mlir::Type;
using mlir::Value;

namespace OpTrait = mlir::OpTrait;

#define GET_OP_CLASSES
#include "tile/plaid_ir/ops.h.inc"

}  // namespace plaid_ir
}  // namespace tile
}  // namespace vertexai
