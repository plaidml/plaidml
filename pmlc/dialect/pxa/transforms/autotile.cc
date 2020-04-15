// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/autotile.h"

#include "pmlc/dialect/pxa/transforms/pass_detail.h"

namespace pmlc::dialect::pxa {

std::vector<int64_t> PowerOfTwoGenerator::operator()(int64_t range) {
  std::vector<int64_t> out;
  for (int64_t r = 1; r <= range; r *= 2) {
    out.push_back(r);
  }
  return out;
}

std::vector<int64_t> EvenTilingGenerator::operator()(int64_t range) {
  std::vector<int64_t> out;
  // TODO: Something less naive: i.e. factor with sieve and then produce
  // divisors via that.  This is not as bad as one might imagine, since
  // generator set is cached in autotile.
  for (int64_t r = 1; r <= range; r++) {
    if (range % r != 0) {
      continue;
    }
    out.push_back(r);
  }
  return out;
}

struct AutoTileExamplePass : public AutoTileExampleBase<AutoTileExamplePass> {
  void runOnFunction() final {
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

std::unique_ptr<mlir::Pass> createAutoTileExamplePass() {
  return std::make_unique<AutoTileExamplePass>();
}

} // namespace pmlc::dialect::pxa
