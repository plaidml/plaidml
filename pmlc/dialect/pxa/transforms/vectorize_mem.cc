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
//    %3 = affine.parallel (%arg2) = (%arg1) to (%arg1 + 8) {
//      %4 = subi %arg2, %arg1 : index
//      %5 = vector.extract_map %2[%4 : 8] : vector<128xf32> to vector<16xf32>
//      %6 = pxa.reduce assign
//      affine.yield %6
//    }
//  }
//
// Sample regarding the vector_reduce assign op:
// Orginal Op:
//  %1 = affine.parallel (%arg1) = (0) to (16) reduce ("assign") {
//    %2 = pxa.load
//    %3 = pxa.reduce assign %2, %arg0[%arg1, 0] : memref<16x16xf32>,
//    vector<16xf32> affine.yield %3
//  }
//
// Modified Op:
//  %1 = affine.parallel (%arg1) = (0) to (16) step (8) {
//    %2 = alloc() : memref<128xf32>
//    %3 = affine.parallel (%arg2) = (%arg1) to (%arg1 + 8) {
//      %4 = pxa.load
//      %5 = subi %arg2, %arg1 : index
//      %6 = vector.insert_map %4, %5, 8 : vector<16xf32> to vector<128xf32>
//      %7 = pxa.vector_reduce assign %6, %2[%c0_1] : memref<128xf32>,
//      vector<128xf32> affine.yield %7
//    }
//    %4 = pxa.vector_load %2[%c0_1] : memref<128xf32>, vector<128xf32>
//    %5 = pxa.vector_reduce assign %4, %arg1[%arg1, 0] : memref<16x16xf32>,
//    vector<128xf32> affine.yield %5
//  }

namespace pmlc::dialect::pxa {

namespace {

struct VectorizeMemImpl {
  VectorizeMemImpl(AffineParallelOp loopOp,
                   std::list<Operation *> &globalAllocList)
      : loopOp(loopOp), globalAllocList(globalAllocList) {}

  struct VectorizeMemOpsPlan {
    // List of vector load\reduce ops that passed the checks for vectorization
    std::list<Operation *> memOps;
    // Index of the loop from which the mem op will be extracted and vectorized
    BlockArgument blockArg;
    // Subgroup size equals vector size of the vector op before modification
    int64_t subgroupSize = 0;
    // Loop size from which the extraction will take place
    int64_t loopVectorSize = 0;
    // Tile size in case there is a need for tiling
    int64_t tileSize = 0;
    // Index of the loop after tiling
    Value tiledBlockArg;
  };

  // Apply checks for vector ops, qualify those for vectorization
  template <typename T>
  void checkMemOp(T memOp) {
    auto defOp = memOp.getMemRef().getDefiningOp();
    // Check if vector_load reads from global memory. Currently it is not
    // allowed to use block ops on in-kernel memory
    if (defOp && std::find(globalAllocList.begin(), globalAllocList.end(),
                           defOp) == globalAllocList.end()) {
      return;
    }
    IVLOG(3, "Mem Op: " << debugString(*memOp));

    // Get strides
    auto maybeSI = computeStrideInfo(memOp);
    if (!maybeSI) {
      return;
    }
    IVLOG(3, "StrideInfo: " << debugString(*maybeSI));

    // Check if read op uses parallel op IV for a single dimension
    for (auto ba : loopOp.getIVs()) {
      if (maybeSI->strides.count(ba)) {
        orgBlockArgs.push_back(ba);
      }
    }

    // Check if block number is not empty,
    // In case it is bigger than one we will handle this upon tiling
    if (orgBlockArgs.empty()) {
      return;
    }

    // Get the size of the considered dimension
    auto ranges = loopOp.getConstantRanges();
    if (!ranges) {
      return;
    }

    auto argNum = orgBlockArgs[orgBlockArgs.size() - 1].getArgNumber();
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
    auto vecShape = memOp.getVectorType().getShape();
    if (vecShape.size() != 1)
      return;
    auto subgroupSize = vecShape[0];

    // Check if data allows for the vectorized block read. Right now it is
    // modelled as an additional dimension that is being added on reaorder
    // layout pass. Thie minimum number of dimensions is 2.
    auto memrefTypeShape = memOp.getMemRefType().getShape();
    if (memrefTypeShape.size() < 2)
      return;

    // Verify if our loop size matches the dimension size.
    if (memrefTypeShape[memrefTypeShape.size() - 2] != loopVectorSize)
      return;

    // Last checks for vectorReduceOp
    auto vectorReduceOp =
        dyn_cast_or_null<PxaVectorReduceOp>(memOp.getOperation());
    if (vectorReduceOp &&
        (vectorReduceOp.getAgg() != AtomicRMWKind::assign ||
         loopOp.results().size() != 1 ||
         (tileSize == loopVectorSize &&
          loopOp.getParentOfType<AffineParallelOp>() &&
          loopOp.getParentOfType<AffineParallelOp>().getResult(0).getType() !=
              loopOp.getResult(0).getType()))) {
      return;
    }

    // Make sure we consider only ops with the same parameters.
    // Fill the values for first read op, the others should be the same
    // as previous ones to be valid.
    if ((!memOpsPlan.loopVectorSize && !memOpsPlan.subgroupSize &&
         !memOpsPlan.tileSize) ||
        (memOpsPlan.blockArg == orgBlockArgs[orgBlockArgs.size() - 1] &&
         memOpsPlan.loopVectorSize == loopVectorSize &&
         memOpsPlan.subgroupSize == subgroupSize &&
         memOpsPlan.tileSize == tileSize)) {
      memOpsPlan.memOps.push_back(memOp);
      memOpsPlan.blockArg = orgBlockArgs[orgBlockArgs.size() - 1];
      memOpsPlan.loopVectorSize = loopVectorSize;
      memOpsPlan.subgroupSize = subgroupSize;
      memOpsPlan.tileSize = tileSize;
    }
  }

  LogicalResult getWritesAndReads() {
    // Starting with the vector load ops, as we mostly care about
    // reads optimization that are more costly than writes
    for (auto vectorLoadOp :
         llvm::make_early_inc_range(loopOp.getOps<PxaVectorLoadOp>())) {
      checkMemOp<PxaVectorLoadOp>(vectorLoadOp);
    }
    for (auto vectorReduceOp :
         llvm::make_early_inc_range(loopOp.getOps<PxaVectorReduceOp>())) {
      checkMemOp<PxaVectorReduceOp>(vectorReduceOp);
    }
    if (!memOpsPlan.memOps.size())
      return failure();
    return success();
  }

  // Helper function to replace argument of the loop in the operation
  // that will be extracted
  void replaceArgInLoop(Operation *memOp) {
    auto blockArg = memOpsPlan.blockArg;

    // Create constant op of value 0, that would replace the actual IV of the
    // loop we are extracting from, in case the loop was not tiled.
    // If it was tiled then it will be removed later on.
    OpBuilder builder(loopOp);
    auto const0 = builder.create<ConstantIndexOp>(loopOp.getLoc(), 0);

    // Replace the IV except for the orginal operation that would become
    // vector.extractelement later. In case of no tiling it would be 0,
    // if tiled then the outer parallel op IV.
    llvm::SmallPtrSet<Operation *, 8> idxNoChange;
    for (auto user : blockArg.getUsers()) {
      if (user != memOp)
        idxNoChange.insert(user);
    }
    blockArg.replaceAllUsesExcept(memOpsPlan.tileSize !=
                                          memOpsPlan.loopVectorSize
                                      ? memOpsPlan.tiledBlockArg
                                      : const0,
                                  idxNoChange);
  }

  void replaceVectorLoad(PxaVectorLoadOp vectorLoad) {
    IVLOG(3, "PxaVectorLoadOp: " << debugString(*vectorLoad));
    auto blockArg = memOpsPlan.blockArg;
    auto loopVectorSize = memOpsPlan.loopVectorSize;
    auto tileSize = memOpsPlan.tileSize;
    auto tiledBlockArg = memOpsPlan.tiledBlockArg;

    replaceArgInLoop(vectorLoad.getOperation());

    OpBuilder builder(loopOp);
    // Create new vector that would be of subgroup size x loop size
    auto vectorType =
        VectorType::get({memOpsPlan.subgroupSize * tileSize},
                        vectorLoad.getVectorType().getElementType());

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
      auto const1 =
          builder.create<SubIOp>(vectorLoad.getLoc(), blockArg, tiledBlockArg);
      const1Result = const1.getResult();
    }
    auto newExtractMapOp = builder.create<vector::ExtractMapOp>(
        vectorLoad.getLoc(), newLoadOp.getResult(),
        tileSize != loopVectorSize ? const1Result : blockArg, tileSize);
    vectorLoad.getResult().replaceAllUsesWith(newExtractMapOp.getResult());
    vectorLoad.erase();
  }

  void replaceVectorReduce(PxaVectorReduceOp vectorReduce) {
    IVLOG(3, "PxaVectorReduceOp: " << debugString(*vectorReduce));
    auto blockArg = memOpsPlan.blockArg;
    auto loopVectorSize = memOpsPlan.loopVectorSize;
    auto tileSize = memOpsPlan.tileSize;
    auto tiledBlockArg = memOpsPlan.tiledBlockArg;

    replaceArgInLoop(vectorReduce.getOperation());

    OpBuilder builder(loopOp);
    // Create new vector that would be of subgroup size x loop size
    auto vectorType =
        VectorType::get({memOpsPlan.subgroupSize * tileSize},
                        vectorReduce.getVectorType().getElementType());

    auto nemMemrefType =
        MemRefType::get(vectorType.getShape(), vectorType.getElementType());
    auto newAllocOp = builder.create<AllocOp>(loopOp.getLoc(), nemMemrefType);
    auto const0 = builder.create<ConstantIndexOp>(loopOp.getLoc(), 0);

    llvm::SmallVector<AffineExpr, 1> expr;
    auto outerDim = builder.getAffineDimExpr(0);
    expr.push_back(outerDim);
    auto emptyMap = AffineMap::get(1, 0, expr, vectorReduce.getContext());

    builder.setInsertionPoint(vectorReduce);
    Value const1Result;
    if (tileSize != loopVectorSize) {
      auto const1 = builder.create<SubIOp>(vectorReduce.getLoc(), blockArg,
                                           tiledBlockArg);
      const1Result = const1.getResult();
    }

    auto newInsertMapOp = builder.create<vector::InsertMapOp>(
        vectorReduce.getLoc(), vectorReduce.vector(),
        tileSize != loopVectorSize ? const1Result : blockArg, tileSize);

    auto results = loopOp.results();
    for (auto res : results) {
      res.setType(nemMemrefType);
    }
    auto newReduceOp = builder.create<PxaVectorReduceOp>(
        vectorReduce.getLoc(), AtomicRMWKind::assign,
        newInsertMapOp.getResult(), newAllocOp, emptyMap,
        ValueRange{const0.getResult()});
    builder.setInsertionPointAfter(loopOp);
    auto newLoadOp = builder.create<PxaVectorLoadOp>(
        vectorReduce.getLoc(), vectorType, newAllocOp, emptyMap,
        ValueRange{const0.getResult()});
    auto newReduceOuterOp = builder.create<PxaVectorReduceOp>(
        vectorReduce.getLoc(), AtomicRMWKind::assign, newLoadOp.getResult(),
        vectorReduce.memref(), vectorReduce.getAffineMap(),
        vectorReduce.idxs());
    for (auto res : results) {
      res.replaceAllUsesWith(newReduceOuterOp.getResult());
    }
    vectorReduce.getResult().replaceAllUsesWith(newReduceOp.getResult());
    vectorReduce.erase();
  }

  void vectorizeOps() {
    // Finished with checks, now perform actual tiling if needed.
    // Do it only once, as we previously checked that tile and vector
    // sizes are the same for all ops.
    // If orgBlockArgs is not 1, the perform tiling anyway, this will
    // help with extracting the mem op from the loop.
    if (memOpsPlan.tileSize != memOpsPlan.loopVectorSize ||
        orgBlockArgs.size() != 1) {
      SmallVector<int64_t, 8> tileSizes;
      // Set tile sizes, care only about the last dim, rest should be set to 1
      // If there are multiple arguments used in the vector op, we still do
      // tiling but only with 1's, as we need additional loop to extract from
      for (int i = 0; i < static_cast<int>(loopOp.getIVs().size()); i++) {
        if (i == static_cast<int>(loopOp.getIVs().size()) - 1 &&
            memOpsPlan.tileSize != 0)
          tileSizes.push_back(memOpsPlan.tileSize);
        else
          tileSizes.push_back(1);
      }
      performTiling(loopOp, tileSizes);
      for (auto newLoopOp : loopOp.getOps<AffineParallelOp>()) {
        memOpsPlan.blockArg = newLoopOp.getIVs()[newLoopOp.getIVs().size() - 1];
        memOpsPlan.tiledBlockArg =
            loopOp.getIVs()[newLoopOp.getIVs().size() - 1];
        loopOp = newLoopOp;
        // Now handle rest of the indices, where tiling was set to 1. It is
        // needed to keep the old arguments instead of the new ones since the
        // vector op will be extracted from the orginal loop.
        if (loopOp.getIVs().size() > 1) {
          for (int i = 0; i < static_cast<int>(loopOp.getIVs().size()) - 1;
               i++) {
            loopOp.getIVs()[i].replaceAllUsesWith(orgBlockArgs[i]);
          }
        }
      }
    }

    // Iterate over vectlo loads and reduce assigns at once, previously
    // we checked that all the parameters for these match.
    for (auto memOp : llvm::make_early_inc_range(memOpsPlan.memOps)) {
      if (auto vectorLoad = dyn_cast<PxaVectorLoadOp>(memOp))
        replaceVectorLoad(vectorLoad);
      else if (auto vectorReduce = dyn_cast<PxaVectorReduceOp>(memOp))
        replaceVectorReduce(vectorReduce);
    }
  }

  // Analyzed loop
  AffineParallelOp loopOp;
  // Orginal block arguments. It is needed for replacing the indices
  SmallVector<BlockArgument, 8> orgBlockArgs;
  // Global memory list
  std::list<Operation *> globalAllocList;
  // Plan for the mem ops vectorization
  VectorizeMemOpsPlan memOpsPlan;
};

void getGlobalMemory(FuncOp f, std::list<Operation *> &globalAllocList) {
  for (auto allocOp : f.getOps<AllocOp>()) {
    globalAllocList.push_back(allocOp.getOperation());
  }
  for (auto parallelOp : f.getOps<AffineParallelOp>()) {
    globalAllocList.push_back(parallelOp.getOperation());
  }
}

struct VectorizeMemPass : public VectorizeMemBase<VectorizeMemPass> {
  void runOnFunction() final {
    FuncOp f = getFunction();

    // Get global memory
    std::list<Operation *> globalAllocList;
    getGlobalMemory(f, globalAllocList);

    f.walk([&](AffineParallelOp loopOp) {
      VectorizeMemImpl vectorizeMemImpl(loopOp, globalAllocList);
      if (failed(vectorizeMemImpl.getWritesAndReads()))
        return;
      vectorizeMemImpl.vectorizeOps();
    });
  }
};

} // namespace

std::unique_ptr<Pass> createVectorizeMemPass() {
  return std::make_unique<VectorizeMemPass>();
}

} // namespace pmlc::dialect::pxa
