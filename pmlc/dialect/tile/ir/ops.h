// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "pmlc/dialect/tile/ir/types.h"
#include "pmlc/dialect/tile/ir/util.h"
#include "pmlc/util/enums.h"

#include "pmlc/dialect/tile/ir/interfaces.h.inc"

namespace pmlc::dialect::tile {

using mlir::IntegerSet;
using mlir::IntegerSetAttr;
using util::AggregationKind;
using util::CombinationKind;

} // namespace pmlc::dialect::tile

#define GET_OP_CLASSES
#include "pmlc/dialect/tile/ir/ops.h.inc"

#include "pmlc/dialect/tile/ir/dialect.h.inc"
