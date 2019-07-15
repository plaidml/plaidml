// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace vertexai {
namespace tile {
namespace plaid_ir {

struct PaddingPass : public mlir::FunctionPass<PaddingPass> {
  void runOnFunction() override;
};

/*
static mlir::PassRegistration<PaddingPass> pass(
        "padding-pass", "Pad convolutions and such");
*/

}  // namespace plaid_ir
}  // namespace tile
}  // namespace vertexai
