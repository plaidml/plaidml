// Copyright 2020, Intel Corporation

#include "pmlc/dialect/pxa/transforms/autotile.h"

namespace pmlc::dialect::pxa {

struct Autotile10Pass : public mlir::FunctionPass<Autotile10Pass> {
  void runOnFunction() override {
    auto func = getFunction();
    FixedTileSizeGenerator always10(10);
    func.walk([&](AffineParallelOp op) {
      auto ranges = op.getConstantRanges();
      if (!ranges) {
        return;
      }
      auto tileSize = FindBestTileSize(always10, DummyCostModel, *ranges);
      if (tileSize.empty()) {
        return;
      }
      performTiling(op, tileSize);
    });
  }
};

static mlir::PassRegistration<Autotile10Pass> pass( //
    "autotile-10", "Tile all dimensions by 10");

} // namespace pmlc::dialect::pxa
