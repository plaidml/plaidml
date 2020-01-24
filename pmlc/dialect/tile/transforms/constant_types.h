// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"

namespace pmlc::dialect::tile {

struct ConstantTypesPass : public mlir::FunctionPass<ConstantTypesPass> {
  void runOnFunction() final;

  mlir::FloatType floatx_;
  mlir::IntegerType intx_;
};

}  // namespace pmlc::dialect::tile
