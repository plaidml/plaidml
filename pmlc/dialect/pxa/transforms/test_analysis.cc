// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/test_analysis.h"
#include "pmlc/dialect/pxa/analysis/strides.h"

namespace pmlc::dialect::pxa {

using namespace llvm; // NOLINT
using namespace mlir; // NOLINT

void TestStrideInfoPass::runOnOperation() {
  auto op = getOperation();
  op->walk([&](AffineLoadOp op) {
    auto info = computeStrideInfo(op);
    llvm::outs() << "stride begin\n";
    if (!info) {
      llvm::outs() << "stride none\n";
    } else {
      llvm::outs() << "offset = " << info->offset << "\n";
      for (auto kvp : info->strides) {
        llvm::outs() << "ba" << kvp.first.getArgNumber() << " = " << kvp.second
                     << "\n";
      }
    }
    llvm::outs() << "stride end\n";
  });
}

} // End namespace pmlc::dialect::pxa
