// Copyright 2021 Intel Corporation

#include "pmlc/dialect/pml/transforms/pass_detail.h"

#include "mlir/Parser.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pml {

struct LoadModulePass : public LoadModuleBase<LoadModulePass> {
  LoadModulePass() = default;

  explicit LoadModulePass(StringRef path) { this->path = path.str(); }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (failed(parseSourceFile(path, module.getBody(), &getContext()))) {
      signalPassFailure();
      return;
    }
  }
};

std::unique_ptr<mlir::Pass> createLoadModulePass() {
  return std::make_unique<LoadModulePass>();
}

std::unique_ptr<mlir::Pass> createLoadModulePass(StringRef path) {
  return std::make_unique<LoadModulePass>(path);
}

} // namespace pmlc::dialect::pml
