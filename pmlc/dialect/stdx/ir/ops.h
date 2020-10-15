// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace pmlc::dialect::stdx {

using mlir::Value;

} // namespace pmlc::dialect::stdx

#define GET_OP_CLASSES
#include "pmlc/dialect/stdx/ir/ops.h.inc"

#include "pmlc/dialect/stdx/ir/dialect.h.inc"
