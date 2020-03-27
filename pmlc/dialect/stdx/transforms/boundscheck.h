// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/Pass/Pass.h"

namespace pmlc::dialect::stdx {

struct BoundsCheckPass : public mlir::FunctionPass<BoundsCheckPass> {
  void runOnFunction() override;

private:
  void generateBoundsChecks(mlir::Operation &op);
};

} // namespace pmlc::dialect::stdx
