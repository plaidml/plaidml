// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Support/DebugStringHelper.h"
#include "pmlc/dialect/pxa/analysis/strides.h"

#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

// TODO: add detailed description here

namespace pmlc::dialect::pxa {

namespace {

// TODO: Maybe move this to a generic utility somewhere
template <typename OpTy, typename... Args>
static OpTy replaceOp(Operation *op, Args &&... args) {
  OpBuilder builder(op);
  auto newOp = builder.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
  op->getResult(0).replaceAllUsesWith(newOp.getResult());
  op->erase();
  return newOp;
}

struct VectorizeMemPass : public VectorizeMemBase<VectorizeMemPass> {
  void runOnFunction() final {
    FuncOp f = getFunction();
    // TODO: generalize this for all read\write ops
    f.walk([&](PxaReadOpInterface loadOp) {
      auto defOp = loadOp.getMemRef().getDefiningOp();
      if (defOp) {
        return;
      }

      // TODO: consider adding case with vload usage also, not only vectorized
      // block read
      auto vectorLoad = dyn_cast<PxaVectorLoadOp>(loadOp.getOperation());
      if (!vectorLoad) {
        return;
      }

      // Check if op is inside AffineParallelOp to extract it from
      auto loopOp = vectorLoad.getParentOfType<AffineParallelOp>();
      if (!loopOp) {
        return;
      }

      IVLOG(3, "Load Op: " << debugString(*vectorLoad));

      // Check strides
      auto maybeSI = computeStrideInfo(vectorLoad);
      if (!maybeSI) {
        return;
      }
      IVLOG(3, "StrideInfo: " << debugString(*maybeSI));

      // TODO: verify if we can do vlock op here based on the strides and
      // dimension we iterate in the parent loop
      SmallVector<BlockArgument, 4> blockArgs;
      for (auto ba : loopOp.getIVs()) {
        if (maybeSI->strides.count(ba)) {
          blockArgs.push_back(ba);
        }
      }

      if (blockArgs.size() != 1) {
        return;
      }

      // TODO: add checks if vector size is valid (2, 4, 8).
      // For the block ops we limit to the 2, 4, 8, but if we
      // consider vload then vec size 16 would be also allowed,
      // at least it is in OpenCL, double check this.
      auto ranges = loopOp.getConstantRanges();
      if (!ranges) {
        return;
      }

      auto argNum = blockArgs[0].getArgNumber();
      auto loopVectorSize = (*ranges)[argNum];
      IVLOG(3, "LoopVecSize: " << loopVectorSize);

      if (!(loopVectorSize == 2 || loopVectorSize == 4 || loopVectorSize == 8))
        return;

      auto vecShape = vectorLoad.getVectorType().getShape();
      // Accept only vectors with dim size 1
      if (vecShape.size() != 1)
        return;

      // Create constant op of value 0, that would replace the actual IV of the
      // loop we are extracting from
      OpBuilder builder(loopOp);
      auto const0 = builder.create<ConstantIndexOp>(loopOp.getLoc(), 0);

      // Replace the IV except for the orginal operation that would become
      // vector.extractelement later
      llvm::SmallPtrSet<Operation *, 8> idxNoChange;
      for (auto user : blockArgs[0].getUsers()) {
        if (user != loadOp)
          idxNoChange.insert(user);
      }
      blockArgs[0].replaceAllUsesExcept(const0, idxNoChange);

      // Create new vector that would be of orginal size x loop size
      auto vectorType =
          VectorType::get({vecShape[0] * loopVectorSize},
                          vectorLoad.getVectorType().getElementType());

      // Create new vector load, extracted from the orginal loop
      auto newLoadOp = builder.create<PxaVectorLoadOp>(
          loopOp.getLoc(), vectorType, vectorLoad.getMemRef(),
          vectorLoad.getAffineMap(), vectorLoad.getMapOperands());

      // TODO: The below commented code implements additional temporary vector
      // in case we would like to put few loads to single vector.
      // auto nemMemrefType = MemRefType::get({1}, vectorType);
      // auto newAllocOp = builder.create<AllocOp>(loopOp.getLoc(),
      // nemMemrefType); SmallVector<Value, 4> indices;
      // indices.push_back(const0.getResult());
      // auto newStoreOp =
      // builder.create<vector::TransferWriteOp>(loopOp.getLoc(),
      // newLoadOp.getResult(), newAllocOp.getResult(), indices); auto
      // newLoadOp2 =
      // builder.create<vector::TransferReadOp>(vectorLoad.getLoc(), vectorType,
      // newStoreOp.memref(), indices);

      // Replace original op with extract map.
      // Use extract map instead of extract element as we do not extract
      // scalar but vector of size equal to subgroup
      replaceOp<vector::ExtractMapOp>(vectorLoad, newLoadOp.getResult(),
                                      blockArgs[0], loopVectorSize);
    });
  }
};

} // namespace

std::unique_ptr<Pass> createVectorizeMemPass() {
  return std::make_unique<VectorizeMemPass>();
}

} // namespace pmlc::dialect::pxa
