// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/test_analysis.h"

#include <map>
#include <string>

#include "mlir/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "pmlc/dialect/pxa/analysis/strides.h"

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
}

void TestStrideInfoPass::runOnOperation() {
  auto op = getOperation();
  op->walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<AffineLoadOp>([](auto op) { printStrideInfo(op); })
        .Case<AffineStoreOp>([](auto op) { printStrideInfo(op); })
        .Case<AffineReduceOp>([](auto op) { printStrideInfo(op); });
  });
}

} // namespace pmlc::dialect::pxa
