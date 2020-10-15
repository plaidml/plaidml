// Copyright 2019, Intel Corporation

#pragma once

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "pmlc/dialect/eltwise/ir/types.h"
#include "pmlc/dialect/eltwise/ir/util.h"
#include "pmlc/util/interfaces.h"

#define GET_OP_CLASSES
#include "pmlc/dialect/eltwise/ir/ops.h.inc"

#include "pmlc/dialect/eltwise/ir/dialect.h.inc"
