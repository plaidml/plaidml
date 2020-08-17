// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

namespace mlir {

Value createIdentity(OpBuilder &builder, Location &loc, AtomicRMWKind agg,
                     Type type);

} // namespace mlir
