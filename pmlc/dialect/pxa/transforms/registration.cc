// Copyright 2020, Intel Corporation

#include "pmlc/dialect/pxa/transforms/autotile.h"

namespace pmlc::dialect::pxa {

struct Autotile10Pass : public mlir::FunctionPass<Autotile10Pass> {
  void runOnFunction() override {
    auto func = this->getFunction();
    FixedTileSizeGenerator always10(10);
    func.walk([&](AffineParallelOp op) {
      llvm::SmallVector<int64_t, 8> ranges;
      if (!op.getConstantRanges(ranges)) {
        return;
      }
      auto tileSize = FindBestTileSize(always10, DummyCostModel, ranges);
      if (tileSize.size() == 0) {
        return;
      }
      Tile(op, tileSize);
    });
  }
};

static mlir::PassRegistration<Autotile10Pass> pass(  //
    "autotile-10", "Tile all dimensions by 10");

}  // namespace pmlc::dialect::pxa
