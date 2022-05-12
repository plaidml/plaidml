// Copyright 2021 Intel Corporation

#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "pmlc/dialect/linalgx/ir/ops.h.inc"

#include "pmlc/dialect/linalgx/ir/dialect.h.inc"
