// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "pmlc/dialect/stdx/ir/ops.h.inc"

#include "pmlc/dialect/stdx/ir/dialect.h.inc"
