// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "pmlc/dialect/comp/ir/interfaces.h"
#include "pmlc/dialect/comp/ir/types.h"

#define GET_OP_CLASSES
#include "pmlc/dialect/comp/ir/ops.h.inc"

#include "pmlc/dialect/comp/ir/dialect.h.inc"
