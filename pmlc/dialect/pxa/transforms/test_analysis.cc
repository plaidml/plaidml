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

static std::string getUniqueName(Block *ref, BlockArgument arg) {
  unsigned reverseDepth = 0;
  while (arg.getOwner() != ref) {
    ref = ref->getParentOp()->getBlock();
    reverseDepth++;
  }
  return llvm::formatv("^bb{0}:%arg{1}", reverseDepth, arg.getArgNumber())
      .str();
}

template <typename T>
static void printStrideInfo(T op) {
  auto info = computeStrideInfo(op);
  llvm::outs() << "stride begin\n";
  if (!info) {
    llvm::outs() << "stride none\n";
  } else {
    llvm::outs() << "offset = " << info->offset << "\n";
    std::map<std::string, unsigned> ordered;
    auto block = op.getOperation()->getBlock();
    for (auto kvp : info->strides) {
      ordered.emplace(getUniqueName(block, kvp.first), kvp.second);
    }
    for (auto kvp : ordered) {
      llvm::outs() << kvp.first << " = " << kvp.second << "\n";
    }
  }
  llvm::outs() << "stride end\n";
  llvm::outs().flush();
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
