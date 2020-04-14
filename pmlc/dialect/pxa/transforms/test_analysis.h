// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/Pass/Pass.h"

namespace pmlc::dialect::pxa {

struct TestStrideInfoPass
    : public mlir::PassWrapper<TestStrideInfoPass, mlir::OperationPass<void>> {
  void runOnOperation() override;
};

} // namespace pmlc::dialect::pxa
