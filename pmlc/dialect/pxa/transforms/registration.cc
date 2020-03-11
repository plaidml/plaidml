// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/autotile.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/pxa/transforms/test_analysis.h"

namespace pmlc::dialect::pxa {

struct Autotile10Pass : public mlir::FunctionPass<Autotile10Pass> {
  void runOnFunction() override {
    auto func = getFunction();
    FixedTileSizeGenerator always10(10);
    func.walk([&](mlir::AffineParallelOp op) {
      auto ranges = op.getConstantRanges();
      if (!ranges) {
        return;
      }
      auto tileSize = findBestTileSize(always10, DummyCostModel, *ranges);
      if (tileSize.empty()) {
        return;
      }
      performTiling(op, tileSize);
    });
  }
};

static mlir::PassRegistration<Autotile10Pass>
    regAutotile10("autotile-10", "Tile all dimensions by 10");

static mlir::PassRegistration<TestStrideInfoPass>
    regTestStrideInfo("test-stride-info",
                      "Report stride data for all loads/stores for unit tests");

static mlir::PassPipelineRegistration<StencilPassOptions>
    stencil_pass("affine-stencil",
                 "Stencil the inputs to create \"micro\" GEMM operations.",
                 createStencilPass);

} // namespace pmlc::dialect::pxa
