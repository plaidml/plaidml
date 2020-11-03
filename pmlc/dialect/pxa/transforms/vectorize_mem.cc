// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Support/DebugStringHelper.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/transforms/tile.h"

#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

// This pass extracts the block read and write operations from the
// AffineParallel loop of certain size and conditions and vectorizes it. Instead
// of having the vector_load or vector_reduce assign that operates on global
// memory inside the loop, we leverage on the vectorized block ops HW
// capability, to perform less reads and extract data from vector inside the
// actual loop. This pass currently assumes using of intel block ops, and gets
// most benefit after moving to block formats in layout reorder pass. The new
// vector size is the multiplication of the subgroup_size and block op vector
// size, and gets reduced upon devectorization and adding actual block ops. The
// new ExtractMapOp cannot be used as is, it gets lowered to
// vector.extractelement in the further passes.
//
// Sample:
// Orginal Op:
//  %1 = affine.parallel (%arg1) = (0) to (16) reduce ("assign") {
//    %2 = pxa.vector_load %arg0[%arg1, 0] : memref<16x16xf32>, vector<16xf32>
//    %3 = pxa.reduce assign
//    affine.yield %3
//  }
//
// Modified Op:
//  %1 = affine.parallel (%arg1) = (0) to (16) step (8) {
//    %2 = pxa.vector_load %arg0[%arg1, 0] : memref<16x16xf32>, vector<128xf32>
//    %3 = affine.parallel (%arg2) = (%arg1) to (%arg1 + 8)
//      %4 = subi %arg2, %arg1 : index
//      %5 = vector.extract_map %2[%4 : 8] : vector<128xf32> to vector<16xf32>
//      %6 = pxa.reduce assign
//      affine.yield %6
//    }
//  }

namespace pmlc::dialect::pxa {

namespace {

void getGlobalMemory(FuncOp f, std::list<Operation *> &globalAllocList) {
  for (auto allocOp : f.getOps<AllocOp>()) {
    globalAllocList.push_back(allocOp.getOperation());
  }
  for (auto parallelOp : f.getOps<AffineParallelOp>()) {
    globalAllocList.push_back(parallelOp.getOperation());
  }
}

void vectorizeReads(AffineParallelOp loopOp,
                    std::list<Operation *> &globalAllocList) {
  for (auto loadOp :
       llvm::make_early_inc_range(loopOp.getOps<PxaReadOpInterface>())) {
    auto defOp = loadOp.getMemRef().getDefiningOp();

    // TODO: consider adding case with vload usage also, right now only support
    // vector_loads
    auto vectorLoad = dyn_cast<PxaVectorLoadOp>(loadOp.getOperation());
    if (!vectorLoad) {
      return;
    }

    // Check if vector_load reads from global memory. Currently it is not
    // allowed to use block ops on in-kernel memory
    if (defOp && std::find(globalAllocList.begin(), globalAllocList.end(),
                           defOp) == globalAllocList.end()) {
      return;
    }
    IVLOG(3, "Load Op: " << debugString(*vectorLoad));

    // Get strides
    auto maybeSI = computeStrideInfo(vectorLoad);
    if (!maybeSI) {
      return;
    }
    IVLOG(3, "StrideInfo: " << debugString(*maybeSI));

    // Check if read op uses parallel op IV for a single dimension
    auto strideVal = 0;
    SmallVector<BlockArgument, 4> blockArgs;
    for (auto ba : loopOp.getIVs()) {
      if (maybeSI->strides.count(ba)) {
        strideVal = maybeSI->strides.find(ba)->second;
        blockArgs.push_back(ba);
      }
    }

    if (blockArgs.size() != 1) {
      return;
    }

    // Get the size of the considered dimension
    auto ranges = loopOp.getConstantRanges();
    if (!ranges) {
      return;
    }

    auto argNum = blockArgs[0].getArgNumber();
    auto loopVectorSize = (*ranges)[argNum];
    IVLOG(3, "LoopVecSize: " << loopVectorSize);

    // Check if loop size is a multiple of allowed vector size
    // for the vectorized block op. Right now only 8, 4, 2 sizes are
    // allowed for this extension. If size is bigger than vector size
    // then there would be needed additional tiling
    auto tileSize = 0;
    SmallVector<int64_t, 4> allowedVecSizes({8, 4, 2});
    for (auto allowedVecSize : allowedVecSizes) {
      if (loopVectorSize % allowedVecSize == 0) {
        tileSize = allowedVecSize;
        break;
      }
    }
    if (!tileSize)
      return;

    // Get the current vector size for vector_load op, this is modelled
    // as subgroup size. With this pass it will be expanded to
    // mem op vector size x subgroup size.
    auto vecShape = vectorLoad.getVectorType().getShape();
    if (vecShape.size() != 1)
      return;
    auto subgroupSize = vecShape[0];

    // Check if data allows for the vectorized block read. Right now it is
    // modelled as an additional dimension that is being added on reaorder
    // layout pass. Thie minimum number of dimensions is 2.
    auto memrefTypeShape = vectorLoad.getMemRefType().getShape();
    if (memrefTypeShape.size() < 2)
      return;

    // Verify if our loop size matches the dimension size.
    if (memrefTypeShape[memrefTypeShape.size() - 2] != loopVectorSize)
      return;

    // Finished with checks, now perform actual tiling if needed
    BlockArgument tiledBlockArg;
    if (tileSize != loopVectorSize) {
      performTiling(loopOp, llvm::ArrayRef<int64_t>({tileSize}));
      for (auto newLoopOp : loopOp.getOps<AffineParallelOp>()) {
        blockArgs[0] = newLoopOp.getIVs()[0];
        tiledBlockArg = loopOp.getIVs()[0];
        loopOp = newLoopOp;
      }
    }

    // Create constant op of value 0, that would replace the actual IV of the
    // loop we are extracting from, in case the loop was not tiled.
    // If it was tiled then it will be removed later on.
    OpBuilder builder(loopOp);
    auto const0 = builder.create<ConstantIndexOp>(loopOp.getLoc(), 0);

    // Replace the IV except for the orginal operation that would become
    // vector.extractelement later. In case of no tiling it would be 0,
    // if tiled then the outer parallel op IV.
    llvm::SmallPtrSet<Operation *, 8> idxNoChange;
    for (auto user : blockArgs[0].getUsers()) {
      if (user != loadOp)
        idxNoChange.insert(user);
    }
    blockArgs[0].replaceAllUsesExcept(
        tileSize != loopVectorSize ? dyn_cast<Value>(tiledBlockArg) : const0,
        idxNoChange);

    // Create new vector that would be of subgroup size x loop size
    auto vectorType = VectorType::get(
        {subgroupSize * tileSize}, vectorLoad.getVectorType().getElementType());

    // Create new vector load with expanded vector size.
    // This op is extracted from the orginal loop.
    auto newLoadOp = builder.create<PxaVectorLoadOp>(
        loopOp.getLoc(), vectorType, vectorLoad.getMemRef(),
        vectorLoad.getAffineMap(), vectorLoad.getMapOperands());

    // Replace original op with extract map. Use extract map instead of extract
    // element as we do not extract scalar but vector of size equal to subgroup.
    // In case we needed tiling, additional fix to the index computation needed
    // to subtract the outer loop IV.
    builder.setInsertionPoint(vectorLoad);
    Value const1Result;
    if (tileSize != loopVectorSize) {
      auto const1 = builder.create<SubIOp>(vectorLoad.getLoc(), blockArgs[0],
                                           tiledBlockArg);
      const1Result = const1.getResult();
    }
    auto newExtractMapOp = builder.create<vector::ExtractMapOp>(
        vectorLoad.getLoc(), newLoadOp.getResult(),
        tileSize != loopVectorSize ? const1Result : blockArgs[0], tileSize);
    vectorLoad.getResult().replaceAllUsesWith(newExtractMapOp.getResult());
    vectorLoad.erase();
  }
}

struct VectorizeMemPass : public VectorizeMemBase<VectorizeMemPass> {
  void runOnFunction() final {
    FuncOp f = getFunction();

    // Get global memory
    std::list<Operation *> globalAllocList;
    getGlobalMemory(f, globalAllocList);

    f.walk([&](AffineParallelOp loopOp) {
      vectorizeReads(loopOp, globalAllocList);
    });
  }
};

} // namespace

std::unique_ptr<Pass> createVectorizeMemPass() {
  return std::make_unique<VectorizeMemPass>();
}

} // namespace pmlc::dialect::pxa
