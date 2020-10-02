// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::pxa {

namespace {

using WriteRead = std::pair<PxaReduceOp, PxaLoadOp>;
using WriteWrite = std::pair<PxaReduceOp, PxaReduceOp>;

struct FusionInfo {
  struct AffineParallelInfo {
    // The affine.parallel op itself.
    AffineParallelOp op;
    // The loop sizes for the corresponding affine.parallel op.
    SmallVector<int64_t, 4> sizes;
    // The tile for the corresponding affine.parallel.op (pre-fusion).
    SmallVector<int64_t, 8> tile;
  };

  AffineParallelInfo aInfo;
  AffineParallelInfo bInfo;
  // The load and store ops
  SmallVector<WriteRead, 4> readAfterWrites;
  SmallVector<WriteWrite, 4> writeAfterWrites;
  // Current state (whether we have a plan or not)
  bool hasPlan;
  // Fuse the ops with exactly matched idxs
  bool exactlyMatch;
  // Specifies the mapping from A's index space into B's index space (post
  // tiling)
  DenseMap<BlockArgument, BlockArgument> aToB;
  DenseMap<BlockArgument, BlockArgument> bToA;
  // Over-fusion prevention parameter
  int64_t memoryActivityThreshold;

  FusionInfo(AffineParallelOp aBand, AffineParallelOp bBand,
             int64_t memoryActivityThreshold, bool exactlyMatch)
      : aInfo{aBand}, bInfo{bBand}, hasPlan(false), exactlyMatch(exactlyMatch),
        memoryActivityThreshold(memoryActivityThreshold) {}

  // Helper method to find the original source write of a state update.
  static PxaReduceOp findSourceWrite(Value val) {
    auto opRes = val.dyn_cast<OpResult>();
    if (auto op = dyn_cast_or_null<PxaReduceOp>(opRes.getOwner())) {
      return op;
    }
    if (auto op = dyn_cast<AffineParallelOp>(opRes.getOwner())) {
      auto retOp = cast<AffineYieldOp>(op.getBody()->getTerminator());
      return findSourceWrite(retOp.getOperand(opRes.getResultNumber()));
    }
    return nullptr;
  }

  // Helper method to remove elements from a stride info that are not part of
  // a specific block's arguments.
  static void cleanStrideInfo(Block *block, StrideInfo &si) {
    for (auto &pair : make_early_inc_range(si.strides)) {
      if (pair.first.getOwner() != block) {
        si.strides.erase(pair.first);
      }
    }
  }

  // Helper to get a clean version of the strides for a specific op (or fail)
  template <typename OpA>
  static bool getStrides(SmallVectorImpl<StrideInfo> &out, OpA op,
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

    IVLOG(3, "considerPlan>");
    IVLOG(3, "  A: " << debugString(*opA));
    IVLOG(3, "  B: " << debugString(*opB));

    aToB.clear();
    bToA.clear();

    // Extract the per-dimension strides for the two ops
    SmallVector<StrideInfo, 4> stridesA;
    if (!getStrides(stridesA, opA, aInfo.op))
      return false;
    SmallVector<StrideInfo, 4> stridesB;
    if (!getStrides(stridesB, opB, bInfo.op))
      return false;

    assert(stridesA.size() == stridesB.size() &&
           "Fusion ops should read/write the same memref and thus have the "
           "same rank");
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
      // If there are multiple indexes, give up
      IVLOG(3, "sa: " << debugString(sa));
      IVLOG(3, "sb: " << debugString(sb));
      if (sa.strides.size() != 1 || sb.strides.size() != 1) {
        opB.emitRemark("Failed to fuse with def due to multiple indexes, i = ")
            << i;
        return false;
      }

      // Extract the details we care about
      BlockArgument argA = sa.strides.begin()->first;
      int64_t mulA = sa.strides.begin()->second;
      int64_t sizeA = aInfo.sizes[argA.getArgNumber()];
      BlockArgument argB = sb.strides.begin()->first;
      int64_t mulB = sb.strides.begin()->second;
      int64_t sizeB = bInfo.sizes[argB.getArgNumber()];

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
          aInfo.op.lowerBoundsMap().getSubMap({argA.getArgNumber()}),
          aInfo.op.getLowerBoundsOperands());
      AffineValueMap lowerB(
          bInfo.op.lowerBoundsMap().getSubMap({argB.getArgNumber()}),
          bInfo.op.getLowerBoundsOperands());
      AffineValueMap diff;
      AffineValueMap::difference(lowerA, lowerB, &diff);
      if (!diff.getAffineMap().isSingleConstant() ||
          diff.getAffineMap().getSingleConstantResult() != 0) {
        bInfo.op.emitRemark("Lower bounds mismatch");
        return false;
      }

      IVLOG(3, "Mapping arg " << argA.getArgNumber() << " to "
                              << argB.getArgNumber());
      if (aToB.count(argA) || bToA.count(argB)) {
        IVLOG(3, "Failed, aToB.count(" << argA.getArgNumber()
                                       << ") = " << aToB.count(argA));
        IVLOG(3, "Failed, bToA.count(" << argB.getArgNumber()
                                       << ") = " << bToA.count(argB));
        bInfo.op.emitRemark("Mapping is not 1 to 1");
        return false;
      }
      aToB[argA] = argB;
      bToA[argB] = argA;
    }

    if (aToB.size() == 0) {
      bInfo.op.emitRemark("No index matches");
      return false;
    }

    if (exactlyMatch && (aToB.size() != aInfo.sizes.size() || bToA.size() != bInfo.sizes.size())) {
      bInfo.op.emitRemark("Loops do not match exactly.");
      return false;
    }

    // Over-fusion prevention:
    // Compute the amount of memory activity, defined as the amount of memory
    // allocated, read, and written during the course of the two affine.parallel
    // ops to be merged.
    // Prevent fusion when the amount of memory activity exceeds a user-defined
    // threshold.
    if (memoryActivityThreshold) {
      auto memoryActivity = computeMemoryActivity();
      if (memoryActivity > memoryActivityThreshold) {
        return false;
      }
    }

    auto aRap = computeThisRelativeAccess(opA);
    auto bRap = computeThisRelativeAccess(opB);
    auto isAliased = hasPerfectAliasing(*aRap, *bRap, bToA);
    IVLOG(3, "isAliased: " << isAliased);

    for (const auto &raw : readAfterWrites) {
      IVLOG(3, "  RAW: " << debugString(*raw.second));
      auto aRap = computeThisRelativeAccess(raw.first);
      auto bRap = computeThisRelativeAccess(raw.second);
      auto ret = hasPerfectAliasing(*aRap, *bRap, bToA);
      IVLOG(3, "  isAliased: " << ret);
      if (!ret) {
        return false;
      }
    }

    for (const auto &waw : writeAfterWrites) {
      IVLOG(3, "  WAW: " << debugString(*waw.second));
      auto aRap = computeThisRelativeAccess(waw.first);
      auto bRap = computeThisRelativeAccess(waw.second);
      auto ret = hasPerfectAliasing(*aRap, *bRap, bToA);
      IVLOG(3, "  isAliased: " << ret);
      if (!ret) {
        return false;
      }
    }

    hasPlan = true;
    return true;
  }

  Optional<RelativeAccessPattern> computeThisRelativeAccess(Operation *op) {
    return computeRelativeAccess(op, [&](BlockArgument arg) {
      return (aToB.count(arg) || bToA.count(arg)) ? BoundaryRegion::Exterior
                                                  : BoundaryRegion::Interior;
    });
  }

  int64_t computeMemoryActivity() {
    int64_t sum = 0;

    auto computeMemoryActivityForOp = [&](Operation *op) {
      if (auto allocOp = dyn_cast<AllocOp>(op)) {
        auto bytes = util::getByteSize(allocOp.getType());
        IVLOG(3, "op: " << debugString(*op) << ", bytes: " << bytes);
        sum += bytes;
      }
      auto relAccess = computeThisRelativeAccess(op);
      if (!relAccess)
        return;
      auto bytes = relAccess->totalInnerBytes();
      IVLOG(3, "op: " << debugString(*op) << ", bytes: " << bytes);
      sum += bytes;
    };

    aInfo.op.walk(computeMemoryActivityForOp);
    bInfo.op.walk(computeMemoryActivityForOp);

    IVLOG(3, "memoryActivity: " << sum);
    return sum;
  }

  bool computeFusion() {
    // Get initial information setup
    auto rangesA = aInfo.op.getConstantRanges();
    if (!rangesA) {
      aInfo.op.emitRemark("Op does not have constant ranges");
      return false;
    }
    std::swap(aInfo.sizes, *rangesA);
    auto rangesB = bInfo.op.getConstantRanges();
    if (!rangesB) {
      bInfo.op.emitRemark("Op does not have constant ranges");
      return false;
    }
    std::swap(bInfo.sizes, *rangesB);
    // First, we find all the write/read and write/write pairs, where block A
    // writes to a value that block B reads from or writes into.
    IVLOG(3, "Collecting read/write information");
    // For each output from loop
    for (OpResult result : aInfo.op.results()) {
      // Find the source write
      auto write = findSourceWrite(result);
      // If it's not a proper affine reduce, give up
      if (!write) {
        aInfo.op.emitRemark("Not all results can be traced to writes");
        return false;
      }
      // For each use of the write:
      for (Operation *user : result.getUsers()) {
        // Check if it is inside B, if not, we don't care, check next use.
        if (!bInfo.op.getOperation()->isAncestor(user))
          continue;
        // Now we make sure it's a read or a write, if not, we can't do fusion,
        // bail.
        if (auto read = dyn_cast<PxaLoadOp>(user)) {
          readAfterWrites.emplace_back(write, read);
        } else if (auto write2 = dyn_cast<PxaReduceOp>(user)) {
          writeAfterWrites.emplace_back(write, write2);
        } else {
          user->emitRemark("Op is not a load or reduce");
          return false;
        }
      }
    }
    // For each raw & waw, consider the plan
    for (auto &raw : readAfterWrites) {
      considerPlan(raw.first, raw.second);
    }
    for (auto &waw : writeAfterWrites) {
      considerPlan(waw.first, waw.second);
    }
    return hasPlan;
  }

  AffineParallelOp applyFusion() {
    OpBuilder builder(bInfo.op);
    // The output types of the combined op is the union of the two inputs
    SmallVector<Type, 6> typesC;

    // First we need to find which results op A are used outside of B.  Those
    // results must also be outputs of the merged block.
    for (OpResult result : aInfo.op.getResults()) {
      bool keep = false;
      for (auto &use : result.getUses()) {
        if (!bInfo.op.getOperation()->isAncestor(use.getOwner()))
          keep = true;
      }
      if (keep)
        typesC.push_back(result.getType());
    }
    // All outputs of B are to be kept
    typesC.insert(typesC.end(), bInfo.op.getResultTypes().begin(),
                  bInfo.op.getResultTypes().end());

    // Use A's lower + upper bounds (they must be equivelant to B's due to
    // prior checks).  We output the new indexes in block argument number order.
    SmallVector<std::pair<BlockArgument, BlockArgument>, 4> orderedIVs;
    for (auto &pair : aToB) {
      orderedIVs.push_back(pair);
    }
    std::sort(orderedIVs.begin(), orderedIVs.end(),
              [](const auto &a, const auto &b) {
                return a.first.getArgNumber() < b.first.getArgNumber();
              });
    SmallVector<AffineExpr, 4> lowerExprsC;
    SmallVector<AffineExpr, 4> upperExprsC;
    SmallVector<int64_t, 4> stepsC;
    DenseMap<BlockArgument, size_t> aToNew;
    auto aSteps = aInfo.op.getSteps();
    for (auto &pair : orderedIVs) {
      aToNew[pair.first] = aToNew.size();
      auto idx = pair.first.getArgNumber();
      lowerExprsC.push_back(aInfo.op.lowerBoundsMap().getResult(idx));
      upperExprsC.push_back(aInfo.op.upperBoundsMap().getResult(idx));
      stepsC.push_back(aSteps[idx]);
    }
    // Compute B mappings to new
    DenseMap<BlockArgument, size_t> bToNew;
    for (auto &pair : bToA) {
      bToNew[pair.first] = aToNew.lookup(pair.second);
    }

    // Construct the new outer parallel op
    auto lowerC = AffineMap::get(aInfo.op.lowerBoundsMap().getNumDims(), 0,
                                 lowerExprsC, aInfo.op.getContext());
    auto upperC = AffineMap::get(aInfo.op.upperBoundsMap().getNumDims(), 0,
                                 upperExprsC, aInfo.op.getContext());
    SmallVector<AtomicRMWKind, 8> reductions(typesC.size(),
                                             AtomicRMWKind::assign);
    auto apC = builder.create<AffineParallelOp>(
        aInfo.op.getLoc(),
        /*resultTypes=*/typesC,
        /*reductions=*/reductions,
        /*lbMap=*/lowerC, /*lbArgs=*/aInfo.op.getLowerBoundsOperands(),
        /*ubMap=*/upperC, /*ubArgs=*/aInfo.op.getUpperBoundsOperands(),
        /*steps=*/stepsC);

    // Move the two parallel for's inside the new op
    aInfo.op.getOperation()->moveBefore(apC.getBody(), apC.getBody()->end());
    bInfo.op.getOperation()->moveBefore(apC.getBody(), apC.getBody()->end());
    // Fixup uses of A's return values.  These uses are either in B (and thus
    // local to C now) or some other op (and thus need to be moved to a return
    // of C).  Basically, for each return, we go over all uses, and adjust to
    // the appropriate new value.  As we go, we add things that escape the the
    // yield output.
    SmallVector<Value, 6> returnVals;
    for (OpResult result : aInfo.op.getResults()) {
      bool keep = false;
      for (OpOperand &use : make_early_inc_range(result.getUses())) {
        // If it's not inside B, swith the use external return value
        if (!bInfo.op.getOperation()->isAncestor(use.getOwner())) {
          keep = true;
          use.set(apC.getResult(returnVals.size()));
        }
      }
      // If we are keeping the value, add to return
      if (keep)
        returnVals.push_back(result);
    }
    // Replace all uses of B's outputs with C's outputs
    for (OpResult result : bInfo.op.getResults()) {
      result.replaceAllUsesWith(apC.getResult(returnVals.size()));
      returnVals.push_back(result);
    }
    // Next we make a new return op for the values that escape this block
    builder.setInsertionPointAfter(bInfo.op);
    builder.create<AffineYieldOp>(bInfo.op.getLoc(), returnVals);

    // Now, we need to update the actual loop bounds for everything
    auto fixupLoops = [&](auto apOp, const auto &toNew) {
      auto origNumArgs = apOp.getBody()->getArguments().size();
      size_t curArgNum = 0;
      SmallVector<AffineExpr, 6> newLowerBounds;
      SmallVector<AffineExpr, 6> newUpperBounds;
      SmallVector<int64_t, 6> newSteps;
      auto apSteps = apOp.getSteps();
      for (size_t i = 0; i < origNumArgs; i++) {
        auto curArg = apOp.getBody()->getArgument(curArgNum);
        auto it = toNew.find(curArg);
        if (it != toNew.end()) {
          curArg.replaceAllUsesWith(apC.getBody()->getArgument(it->second));
          apOp.getBody()->eraseArgument(curArgNum);
        } else {
          newLowerBounds.push_back(apOp.lowerBoundsMap().getResult(i));
          newUpperBounds.push_back(apOp.upperBoundsMap().getResult(i));
          newSteps.push_back(apSteps[i]);
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
    fixupLoops(aInfo.op, aToNew);
    fixupLoops(bInfo.op, bToNew);

    return apC;
  }
};

struct FusionPass : public FusionBase<FusionPass> {
  FusionPass() = default;

  explicit FusionPass(int64_t memoryActivityThreshold, bool exactlyMatch) {
    this->memoryActivityThreshold = memoryActivityThreshold;
    this->exactlyMatch = exactlyMatch;
  }

  // Attempts to fuse two ops if they look good.  Returns the new fused loop
  // (which may be a nullptr if fusion failed).
  AffineParallelOp attemptFusion(AffineParallelOp aBand,
                                 AffineParallelOp bBand) {
    IVLOG(4, "Attempt fusion:\nA:\n"
                 << debugString(*aBand) << "\nB:\n"
                 << debugString(*bBand));
    FusionInfo fusionInfo(aBand, bBand, memoryActivityThreshold.getValue(), exactlyMatch);
    bool canFuse = fusionInfo.computeFusion();
    if (!canFuse) {
      return nullptr;
    }
    IVLOG(3, "Found " << fusionInfo.readAfterWrites.size()
                      << " read after writes");
    IVLOG(3, "Found " << fusionInfo.writeAfterWrites.size()
                      << " write after writes");
    return fusionInfo.applyFusion();
  }

  void runOnFunction() final {
    auto func = getFunction();
    // Autotile only the outermost loops: TODO how should a user specify which
    // blocks to consider?
    auto &block = func.getBody().front();
    // First we 'number' every op
    DenseMap<Operation *, size_t> opOrder;
    for (auto &op : block) {
      opOrder[&op] = opOrder.size();
    }

    for (auto itOp = block.begin(); itOp != block.end();) {
      auto fuseA = dyn_cast<AffineParallelOp>(*itOp);
      // Kick the iterator forward right away so if we end up fusing the op
      // down into a successor, we don't cause issues
      ++itOp;
      // Only consider affine.parallel ops
      if (!fuseA) {
        continue;
      }

      // Find the 'nearest reader' block:  Walk over each output, find any
      // blocks that they output. pick the block closest to the writer.  This
      // block is legal to fuse into, since there are no intermediating
      // dependencies.
      Operation *nearestReader = nullptr;
      for (auto res : fuseA.results()) {
        for (auto user : res.getUsers()) {
          Operation *op = block.findAncestorOpInBlock(*user);
          if (!nearestReader || opOrder[op] < opOrder[nearestReader]) {
            nearestReader = op;
          }
        }
      }
      // Check if the closest reader is also and affine parallel, in which case,
      // attempt to merge
      if (auto fuseB = dyn_cast_or_null<AffineParallelOp>(nearestReader)) {
        bool merge_immediate = (&*itOp == nearestReader);
        // Put the new op at the opOrderition of B
        size_t newOpOrder = opOrder[fuseB.getOperation()];
        // Attempt the actual fusion
        AffineParallelOp newOp = attemptFusion(fuseA, fuseB);
        // If it worked, update positions
        if (newOp) {
          opOrder[newOp.getOperation()] = newOpOrder;
          if (merge_immediate) {
            itOp = Block::iterator(newOp.getOperation());
          }
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createFusionPass(int64_t memoryActivityThreshold, bool exactlyMatch) {
  return std::make_unique<FusionPass>(memoryActivityThreshold, exactlyMatch);
}

} // namespace pmlc::dialect::pxa
