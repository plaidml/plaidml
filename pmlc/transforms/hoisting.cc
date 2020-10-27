// Copyright 2020 Intel Corporation

#include "pmlc/transforms/pass_detail.h"

namespace pmlc::transforms {
namespace {

class HoistingPass final : public HoistingPassBase<HoistingPass> {
public:
  void runOnOperation() final;
};

void HoistingPass::runOnOperation() { return; }

} // namespace

std::unique_ptr<mlir::Pass> createHoistingPass() {
  return std::make_unique<HoistingPass>();
}

} // namespace pmlc::transforms
