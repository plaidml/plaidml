// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/Pass/Pass.h"

namespace pmlc::dialect::pxa {

struct TestStrideInfoPass : public mlir::OperationPass<TestStrideInfoPass> {
  void runOnOperation() override;
};

} // namespace pmlc::dialect::pxa
