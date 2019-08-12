// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "tile/codegen/codegen.pb.h"

namespace pmlc {
namespace dialect {
namespace mir {

struct PaddingPass : public mlir::FunctionPass<PaddingPass> {
  explicit PaddingPass(const vertexai::tile::codegen::proto::MirPadPass& options) : options(options) {}
  void runOnFunction() override;

  vertexai::tile::codegen::proto::MirPadPass options;
};

}  // namespace mir
}  // namespace dialect
}  // namespace pmlc
