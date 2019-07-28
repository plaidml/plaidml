// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace pmlc {
namespace dialect {
namespace mir {

struct PaddingPass : public mlir::FunctionPass<PaddingPass> {
  void runOnFunction() override;
};

/*
static mlir::PassRegistration<PaddingPass> pass(
        "padding-pass", "Pad convolutions and such");
*/

}  // namespace mir
}  // namespace dialect
}  // namespace pmlc
