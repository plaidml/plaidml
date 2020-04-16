// Copyright 2020 Intel Corporation

#include <map>
#include <string>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"

namespace pmlc::dialect::pxa {

using namespace llvm; // NOLINT
using namespace mlir; // NOLINT

template <typename T>
static void printStrideInfo(T op) {
  auto info = computeStrideInfo(op);
  llvm::outs() << "strides: ";
  if (!info) {
    llvm::outs() << "none";
  } else {
    auto block = op.getOperation()->getBlock();
    info->print(llvm::outs(), block);
  }
  llvm::outs() << '\n';
  llvm::outs().flush();
}

struct TestStrideInfoPass : public TestStrideInfoBase<TestStrideInfoPass> {
  void runOnOperation() final {
    auto op = getOperation();
    op->walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .Case<AffineLoadOp>([](auto op) { printStrideInfo(op); })
          .Case<AffineStoreOp>([](auto op) { printStrideInfo(op); })
          .Case<AffineReduceOp>([](auto op) { printStrideInfo(op); });
    });
  }
};

std::unique_ptr<mlir::Pass> createTestStrideInfoPass() {
  return std::make_unique<TestStrideInfoPass>();
}

} // namespace pmlc::dialect::pxa
