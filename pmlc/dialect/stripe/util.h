// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/Function.h"

namespace pmlc {
namespace dialect {
namespace stripe {

/// Wraps function body with a ParallelForOp to represent Stripe's 'main' block.
void createMainParallelFor(mlir::FuncOp funcOp);

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
