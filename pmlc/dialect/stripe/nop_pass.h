// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "tile/codegen/codegen.pb.h"

namespace pmlc {
namespace dialect {
namespace stripe {

struct NopPass : public mlir::FunctionPass<NopPass> {
  explicit NopPass(const vertexai::tile::codegen::proto::MLIR_NopPass& options) {}
  void runOnFunction() override {}
};

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
