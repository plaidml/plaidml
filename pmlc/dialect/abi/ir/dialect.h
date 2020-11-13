// Copyright 2020, Intel Corporation

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace pmlc::dialect::abi {

using mlir::ArrayRef;
using mlir::LogicalResult;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::Region;
using mlir::Value;

namespace OpTrait = mlir::OpTrait;

} // namespace pmlc::dialect::abi

#define GET_OP_CLASSES
#include "pmlc/dialect/abi/ir/ops.h.inc"

#include "pmlc/dialect/abi/ir/dialect.h.inc"
