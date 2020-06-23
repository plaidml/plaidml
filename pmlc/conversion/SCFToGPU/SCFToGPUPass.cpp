//===- SCFToGPUPass.cpp - Convert a loop nest to a GPU kernel -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pmlc/conversion/SCFToGPU/SCFToGPUPass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"

#include "pmlc/conversion/SCFToGPU/PassDetail.h"
#include "pmlc/conversion/SCFToGPU/SCFToGPU.h"

#define PASS_NAME "convert-scf-to-gpu"
#define LOOPOP_TO_GPU_PASS_NAME "convert-loop-op-to-gpu"

using namespace mlir;      // NOLINT
using namespace mlir::scf; // NOLINT

namespace pmlc::conversion::scf_to_gpu {

namespace {
// A pass that traverses top-level loops in the function and converts them to
// GPU launch operations.  Nested launches are not allowed, so this does not
// walk the function recursively to avoid considering nested loops.
struct ForLoopMapper : public ConvertSimpleSCFToGPUBase<ForLoopMapper> {
  ForLoopMapper() = default;
  ForLoopMapper(unsigned numBlockDims, unsigned numThreadDims) {
    this->numBlockDims = numBlockDims;
    this->numThreadDims = numThreadDims;
  }

  void runOnFunction() override {
    for (Operation &op : llvm::make_early_inc_range(getFunction().getOps())) {
      if (auto forOp = dyn_cast<AffineForOp>(&op)) {
        if (failed(convertAffineLoopNestToGPULaunch(forOp, numBlockDims,
                                                    numThreadDims)))
          signalPassFailure();
      } else if (auto forOp = dyn_cast<ForOp>(&op)) {
        if (failed(
                convertLoopNestToGPULaunch(forOp, numBlockDims, numThreadDims)))
          signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
createSimpleSCFToGPUPass(unsigned numBlockDims, unsigned numThreadDims) {
  return std::make_unique<ForLoopMapper>(numBlockDims, numThreadDims);
}

std::unique_ptr<OperationPass<FuncOp>> createSimpleSCFToGPUPass() {
  return std::make_unique<ForLoopMapper>();
}

} // namespace pmlc::conversion::scf_to_gpu
