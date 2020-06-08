// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

namespace pmlc::dialect::pxa {

using mlir::AffineLoadOp;
using mlir::AffineParallelOp;
using mlir::Block;
using mlir::BlockArgument;
using mlir::Operation;
using mlir::StrideInfo;

namespace {

using WriteRead = std::pair<AffineReduceOp, AffineLoadOp>;
using WriteWrite = std::pair<AffineReduceOp, AffineReduceOp>;

struct FusionInfo {
  // The parallel for ops
  AffineParallelOp apA;
  AffineParallelOp apB;
  // The loop sizes for apA and apB
  llvm::SmallVector<int64_t, 4> sizesA;
  llvm::SmallVector<int64_t, 4> sizesB;
  // The load and store ops
  llvm::SmallVector<WriteRead, 4> writeReads;
  llvm::SmallVector<WriteWrite, 4> writeWrites;
  // Current state (whether we have a plan or not)
  bool hasPlan;
  // Tiling's for A + B (pre-fusion)
  llvm::SmallVector<int64_t, 8> tileA;
  llvm::SmallVector<int64_t, 8> tileB;
  // Specifies the mapping from A's index space into B's index space (post
  // tiling)
  llvm::DenseMap<BlockArgument, BlockArgument> aToB;
  llvm::DenseMap<BlockArgument, BlockArgument> bToA;

  FusionInfo(AffineParallelOp apA, AffineParallelOp apB)
      : apA(apA), apB(apB), hasPlan(false) {}

  // Helper method to find the original source write of a state update.
  static AffineReduceOp findSourceWrite(Value val) {
    auto opRes = val.dyn_cast<mlir::OpResult>();
    if (auto op = mlir::dyn_cast_or_null<AffineReduceOp>(opRes.getOwner())) {
      return op;
    }
    if (auto op = mlir::dyn_cast<AffineParallelOp>(opRes.getOwner())) {
      auto retOp =
          mlir::cast<mlir::AffineYieldOp>(op.getBody()->getTerminator());
      return findSourceWrite(retOp.getOperand(opRes.getResultNumber()));
    }
    return AffineReduceOp();
  }

  // Helper method to remove elements from a stride info that are not part of
  // a specific block's arguments.
  static void cleanStrideInfo(Block *block, StrideInfo &si) {
    for (auto it = si.strides.begin(); it != si.strides.end();) {
      auto next = std::next(it);
      if (it->first.getOwner() != block) {
        si.strides.erase(it);
      }
      it = next;
    }
  }

  // Helper to get a clean version of the strides for a specific op (or fail)
  template <typename OpA>
  static bool getStrides(llvm::SmallVectorImpl<StrideInfo> &out, OpA op,
                         AffineParallelOp ap) {
    auto strides = computeStrideInfo(op.getAffineMap(), op.getMapOperands());
    if (!strides) {
      op.emitRemark("Failed to compute strides");
      return false;
    }
    for (auto si : *strides) {
      cleanStrideInfo(ap.getBody(), si);
      out.push_back(si);
    }
    return true;
  }

  template <typename OpA, typename OpB>
  bool considerPlan(OpA opA, OpB opB) {
    // Early exit is we already have a plan
    if (hasPlan)
      return true;

    // Extract the per-dimension strides for the two ops
    llvm::SmallVector<StrideInfo, 4> stridesA;
    llvm::SmallVector<StrideInfo, 4> stridesB;
    if (!getStrides(stridesA, opA, apA))
      return false;
    if (!getStrides(stridesB, opB, apB))
      return false;

    assert(stridesA.size() == stridesB.size());
    // Try to relate the block arguments
    for (size_t i = 0; i < stridesA.size(); i++) {
      const auto &sa = stridesA[i];
      const auto &sb = stridesB[i];
      // If the offsets don't match, bail
      if (sa.offset != sb.offset) {
        opB.emitRemark("Failed to fuse with def due to offsets, i = ") << i;
        return false;
      }
      // If both are empty, nothing to do
      if (sa.strides.size() == 0 && sb.strides.size() == 0)
        continue;
      // TODO: Thses should be fixed by canonicalization
      if (sa.strides.size() == 0 && sb.strides.size() == 1 &&
          sizesB[sb.strides.begin()->first.getArgNumber()] == 1)
        continue;
      if (sb.strides.size() == 0 && sa.strides.size() == 1 &&
          sizesA[sa.strides.begin()->first.getArgNumber()] == 1)
        continue;
      // If there are multiple indexes, give up
      if (sa.strides.size() != 1 || sb.strides.size() != 1) {
        opB.emitRemark("Failed to fuse with def due to multiple indexes, i = ")
            << i;
        return false;
      }

      // Extract the details we care about
      BlockArgument argA = sa.strides.begin()->first;
      int64_t mulA = sa.strides.begin()->second;
      int64_t sizeA = sizesA[argA.getArgNumber()];
      BlockArgument argB = sb.strides.begin()->first;
      int64_t mulB = sb.strides.begin()->second;
      int64_t sizeB = sizesB[argB.getArgNumber()];

      // Fail if the total range of the two arguments doesn't match
      if (mulA * sizeA != mulB * sizeB) {
        opB.emitRemark("Failed to fuse with def due to mismatched ranges, i = ")
            << i << ": " << mulA * sizeA << " vs " << mulB * sizeB;
        return false;
      }

      // Fail if we need to tile for now.  TODO: Implement tiling
      if (mulA != mulB) {
        opB.emitRemark(
            "Failed to fuse with def due to mismatched strides, i = ")
            << i << ": " << mulA << " vs " << mulB;
        return false;
      }

      // Also fail if the AP's don't have the same lower bound
      AffineValueMap lowerA(
          apA.lowerBoundsMap().getSubMap({argA.getArgNumber()}),
          apA.getLowerBoundsOperands());
      AffineValueMap lowerB(
          apB.lowerBoundsMap().getSubMap({argB.getArgNumber()}),
          apB.getLowerBoundsOperands());
      AffineValueMap diff;
      AffineValueMap::difference(lowerA, lowerB, &diff);
      if (!diff.getAffineMap().isSingleConstant() ||
          diff.getAffineMap().getSingleConstantResult() != 0) {
        apB.emitRemark("Lower bounds mismatch");
        return false;
      }

      IVLOG(1, "Mapping arg " << argA.getArgNumber() << " to "
                              << argB.getArgNumber());
      if (aToB.count(argA) || bToA.count(argB)) {
        IVLOG(1, "Failed, aToB.count(" << argA.getArgNumber()
                                       << ") = " << aToB.count(argA));
        IVLOG(1, "Failed, bToA.count(" << argB.getArgNumber()
                                       << ") = " << bToA.count(argB));
        apB.emitRemark("Mapping is not 1 to 1");
        return false;
      }
      aToB[argA] = argB;
      bToA[argB] = argA;
    }
    if (aToB.size() == 0) {
      apB.emitRemark("No index matches");
      return false;
    }
    hasPlan = true;
    return true;
  }

  bool computeFusion() {
    // Get initial information setup
    auto rangesA = apA.getConstantRanges();
    if (!rangesA) {
      apA.emitRemark("Op does not have constant ranges");
      return false;
    }
    std::swap(sizesA, *rangesA);
    auto rangesB = apB.getConstantRanges();
    if (!rangesB) {
      apB.emitRemark("Op does not have constant ranges");
      return false;
    }
    std::swap(sizesB, *rangesB);
    // First, we find all the write/read and write/write pairs, where block A
    // writes to a value that block B reads from or writes into.
    IVLOG(1, "Collectiong read/write information");
    // For each output from loop
    for (auto res : apA.results()) {
      // Find the source write
      auto write = findSourceWrite(res);
      // If it's not a proper affine reduce, give up
      if (!write) {
        apA.emitRemark("Not all results can be traced to writes");
        return false;
      }
      // For each use of the write:
      for (auto user : res.getUsers()) {
        // Check if it is inside B, if not, we don't care, check next use.
        if (!apB.getOperation()->isAncestor(user))
          continue;
        // Now we make sure it's a read or a write, if not, we can't do fusion,
        // bail.
        if (auto read = mlir::dyn_cast<AffineLoadOp>(user)) {
          if (!considerPlan(write, read))
            return false;
          writeReads.emplace_back(write, read);
        } else if (auto write2 = mlir::dyn_cast<AffineReduceOp>(user)) {
          if (!considerPlan(write, write2))
            return false;
          writeWrites.emplace_back(write, write2);
        } else {
          user->emitRemark("Op is not a load or reduce");
          return false;
        }
      }
    }
    return true;
  }

  AffineParallelOp applyFusion() {
    OpBuilder builder(apB);
    // The output types of the combined op is the union of the two inputs
    llvm::SmallVector<mlir::Type, 6> typesC;

    // First we need to find which results op A are used outside of B.  Those
    // results must also be outputs of the merged block.
    for (auto res : apA.getResults()) {
      bool keep = false;
      for (auto &use : res.getUses()) {
        if (!apB.getOperation()->isAncestor(use.getOwner()))
          keep = true;
      }
      if (keep)
        typesC.push_back(res.getType());
    }
    // All outputs of B are to be kept
    typesC.insert(typesC.end(), apB.getResultTypes().begin(),
                  apB.getResultTypes().end());

    // Use apA's lower + upper bounds (they must be equivelant to apB's due to
    // prior checks).  We output the new indexes in map order
    llvm::SmallVector<AffineExpr, 4> lowerExprsC;
    llvm::SmallVector<AffineExpr, 4> upperExprsC;
    llvm::SmallVector<int64_t, 4> stepsC;
    llvm::DenseMap<BlockArgument, size_t> aToNew;
    for (auto &pair : aToB) {
      aToNew[pair.first] = aToNew.size();
      auto idx = pair.first.getArgNumber();
      lowerExprsC.push_back(apA.lowerBoundsMap().getResult(idx));
      upperExprsC.push_back(apA.upperBoundsMap().getResult(idx));
      stepsC.push_back(apA.steps()[idx].cast<IntegerAttr>().getInt());
    }
    // Compute B mappings to new
    llvm::DenseMap<BlockArgument, size_t> bToNew;
    for (auto &pair : bToA) {
      bToNew[pair.first] = aToNew.lookup(pair.second);
    }

    // Construct the new outer parallel op
    auto lowerC = AffineMap::get(apA.lowerBoundsMap().getNumDims(), 0,
                                 lowerExprsC, apA.getContext());
    auto upperC = AffineMap::get(apA.upperBoundsMap().getNumDims(), 0,
                                 upperExprsC, apA.getContext());
    auto apC = builder.create<AffineParallelOp>(
        apA.getLoc(), typesC, lowerC, apA.getLowerBoundsOperands(), upperC,
        apA.getUpperBoundsOperands(), stepsC);

    // Move the two parallel for's inside the new op
    apA.getOperation()->moveBefore(apC.getBody(), apC.getBody()->end());
    apB.getOperation()->moveBefore(apC.getBody(), apC.getBody()->end());
    // Fixup uses of A's  return values.  These uses are either in B (and thus
    // local to C now) or some other op (and thus need to be moved to a return
    // of C).  Basically, for each return, we go over all uses, and adjust to
    // the appropriate new value.  As we go, we add things that escape the the
    // yield output.
    llvm::SmallVector<Value, 6> returnVals;
    for (auto res : apA.getResults()) {
      bool keep = false;
      for (mlir::OpOperand &use : llvm::make_early_inc_range(res.getUses())) {
        // If it's not inside B, swith the use external return value
        if (!apB.getOperation()->isAncestor(use.getOwner())) {
          keep = true;
          use.set(apC.getResult(returnVals.size()));
        }
      }
      // If we are keeping the value, add to return
      if (keep)
        returnVals.push_back(res);
    }
    // Replace all uses of B's outputs with C's outputs
    for (auto res : apB.getResults()) {
      res.replaceAllUsesWith(apC.getResult(returnVals.size()));
      returnVals.push_back(res);
    }
    // Next we make a new return op for the values that escape this block
    builder.setInsertionPointAfter(apB);
    builder.create<AffineYieldOp>(apB.getLoc(), returnVals);

    // Now, we need to update the actual loop bounds for everything
    auto fixupLoops = [&](auto apOp, const auto &toNew) {
      auto origNumArgs = apOp.getBody()->getArguments().size();
      size_t curArgNum = 0;
      llvm::SmallVector<AffineExpr, 6> newLowerBounds;
      llvm::SmallVector<AffineExpr, 6> newUpperBounds;
      llvm::SmallVector<int64_t, 6> newSteps;
      for (size_t i = 0; i < origNumArgs; i++) {
        auto curArg = apOp.getBody()->getArgument(curArgNum);
        auto it = toNew.find(curArg);
        if (it != toNew.end()) {
          curArg.replaceAllUsesWith(apC.getBody()->getArgument(it->second));
          apOp.getBody()->eraseArgument(curArgNum);
        } else {
          newLowerBounds.push_back(apOp.lowerBoundsMap().getResult(i));
          newUpperBounds.push_back(apOp.lowerBoundsMap().getResult(i));
          newSteps.push_back(
              apOp.steps()[i].template cast<IntegerAttr>().getInt());
          curArgNum++;
        }
      }
      auto newLower = AffineMap::get(apOp.lowerBoundsMap().getNumDims(), 0,
                                     newLowerBounds, apOp.getContext());
      auto newUpper = AffineMap::get(apOp.upperBoundsMap().getNumDims(), 0,
                                     newUpperBounds, apOp.getContext());
      apOp.setAttr(AffineParallelOp::getLowerBoundsMapAttrName(),
                   AffineMapAttr::get(newLower));
      apOp.setAttr(AffineParallelOp::getUpperBoundsMapAttrName(),
                   AffineMapAttr::get(newUpper));
      apOp.setAttr(AffineParallelOp::getStepsAttrName(),
                   builder.getI64ArrayAttr(newSteps));
    };
    fixupLoops(apA, aToNew);
    fixupLoops(apB, bToNew);
    return apC;
    // PRINT!
    // mlir::OpPrintingFlags flags;
    // flags.printGenericOpForm();
    // apC.getOperation()->print(llvm::errs(), flags);
  }
};

struct FusionPass : public FusionBase<FusionPass> {
  // Attempts to fuse two ops if they look good.  Returns the new fused loop
  // (which may be a nullptr if fusion failed).
  AffineParallelOp attemptFusion(AffineParallelOp apA, AffineParallelOp apB) {
    FusionInfo fi(apA, apB);
    bool r = fi.computeFusion();
    if (!r) {
      return AffineParallelOp();
    }
    IVLOG(1, "Found " << fi.writeReads.size() << " write/reads");
    IVLOG(1, "Found " << fi.writeWrites.size() << " write/writes");
    return fi.applyFusion();
  }

  void runOnFunction() final {
    auto func = getFunction();
    // Autotile only the outermost loops: TODO how should a user specify which
    // blocks to consider?
    auto &block = func.getBody().front();
    // First we 'number' every op
    llvm::DenseMap<Operation *, size_t> pos;
    for (auto opIt = block.begin(); opIt != block.end(); ++opIt) {
      pos[&*opIt] = pos.size();
    }

    for (auto opIt = block.begin(); opIt != block.end();) {
      // See if the top op is an affine parallel
      auto fuseA = mlir::dyn_cast<AffineParallelOp>(*opIt);
      // Kick the iterator forward right away so if we end up fusing the op
      // down into a successor, we don't cause issues
      ++opIt;
      // If it's not a AF, continue
      if (!fuseA)
        continue;
      // Find the 'nearest reader' block:  Walk over each output, find any
      // blocks that ther output. pick the block closest to the writer.  This
      // block is legal to fuse into, since there are no intermediating
      // dependencies.
      Operation *nearestReader = nullptr;
      for (auto res : fuseA.results()) {
        for (auto user : res.getUsers()) {
          Operation *op = block.findAncestorOpInBlock(*user);
          if (nearestReader == nullptr || pos[op] < pos[nearestReader]) {
            nearestReader = op;
          }
        }
      }
      // Check if the closest reader is also and affine parallel, in which case,
      // attempt to merge
      if (auto fuseB =
              mlir::dyn_cast_or_null<AffineParallelOp>(nearestReader)) {
        bool merge_immediate = (&*opIt == nearestReader);
        // Put the new op at the position of B
        size_t newPos = pos[fuseB.getOperation()];
        // Attempt the actual fusion
        AffineParallelOp newOp = attemptFusion(fuseA, fuseB);
        // If it worked, update positions
        if (newOp) {
          pos[newOp.getOperation()] = newPos;
          if (merge_immediate) {
            opIt = Block::iterator(newOp.getOperation());
          }
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createFusionPass() {
  return std::make_unique<FusionPass>();
}

} // namespace pmlc::dialect::pxa
