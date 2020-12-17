// Copyright 2020, Intel Corporation

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"

using namespace mlir;                // NOLINT
using namespace pmlc::dialect::stdx; // NOLINT

namespace pmlc::dialect::tile {

namespace {

struct ExpandReshapePass : public ExpandReshapeBase<ExpandReshapePass> {
  void runOnFunction() final;
};

void ExpandReshapePass::runOnFunction() {
  return;
}

} // namespace

std::unique_ptr<Pass> createExpandReshapePass() {
  return std::make_unique<ExpandReshapePass>();
}

} // namespace pmlc::dialect::tile
