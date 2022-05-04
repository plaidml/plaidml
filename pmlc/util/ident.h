// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

namespace mlir {

Value createIdentity(OpBuilder &builder, Location loc, arith::AtomicRMWKind agg,
                     Type type);

} // namespace mlir
