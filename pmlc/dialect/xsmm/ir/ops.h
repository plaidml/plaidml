// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "pmlc/dialect/xsmm/ir/ops.h.inc"

#include "pmlc/dialect/xsmm/ir/dialect.h.inc"
