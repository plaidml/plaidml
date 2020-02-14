// Copyright 2020, Intel Corporation

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/util/logging.h"

namespace mlir {

// Multiple all strides/offets by a constant
static void mulStrides(StrideInfo &val, int64_t mul) {
  val.offset *= mul;
  for (auto &kvp : val.strides) {
    kvp.second *= mul;
  }
}

// Add the second stride info into the first
static void addStrides(StrideInfo &val, const StrideInfo &toAdd) {
  val.offset += toAdd.offset;
  for (const auto &kvp : toAdd.strides) {
    val.strides[kvp.first] += kvp.second;
  }
}

static Optional<StrideInfo> computeStrideInfo(AffineParallelOp op,
                                              BlockArgument arg) {
  IVLOG(1, "PFOR");
  // Start at the lower bound, fail early if lower bound fails.
  size_t idx = arg.getArgNumber();
  auto out = computeStrideInfo(op.lowerBoundsMap().getResult(idx),
                               op.getLowerBoundsOperands());
  if (!out)
    return out;

  // Otherwise add current index's contribution.
  // TODO: getStep(size_t) on AffineParallelOp?
  auto stepAttr = op.steps().getValue()[idx];
  out->strides[arg] += stepAttr.cast<IntegerAttr>().getInt();
  return out;
}

static Optional<StrideInfo> computeStrideInfo(AffineForOp op,
                                              BlockArgument arg) {
  // Get lower bound
  auto map = op.getLowerBoundMap();

  // If it's not a simple lower bound, give up.
  if (map.getNumResults() != 1)
    return None;

  // Compute the effect of the lower bound, fail early if needed.
  auto out = computeStrideInfo(op.getLowerBoundMap().getResult(0),
                               op.getLowerBoundOperands());
  if (!out)
    return None;

  // Otherwise add current index's contribution.
  out->strides[arg] += op.getStep();
  return out;
}

Optional<StrideInfo> computeStrideInfo(Value expr) {
  // First, check for a block argument.
  if (auto arg = expr.dyn_cast<BlockArgument>()) {
    // Check the kind of loop we are part of, and dispatch.
    Operation *baseOp = arg.getOwner()->getParentOp();

    if (auto op = dyn_cast<AffineParallelOp>(baseOp))
      return computeStrideInfo(op, arg);

    if (auto op = dyn_cast<AffineForOp>(baseOp))
      return computeStrideInfo(op, arg);

    // Is this an assertable condition?
    return None;
  }

  // Try for the affine apply case
  if (auto op = dyn_cast<AffineApplyOp>(expr.getDefiningOp()))
    return computeStrideInfo(op.getAffineMap().getResult(0),
                             op.getMapOperands());

  return None;
}

Optional<StrideInfo> computeStrideInfo(AffineExpr expr, ValueRange args) {
  // If we are a constant affine expression, it's a simple offset.
  if (auto cexpr = expr.dyn_cast<AffineConstantExpr>()) {
    StrideInfo r;
    r.offset = cexpr.getValue();
    return r;
  }

  // If we are a dim, it's just a Value.
  if (auto dexpr = expr.dyn_cast<AffineDimExpr>())
    return computeStrideInfo(args[dexpr.getPosition()]);

  // Check the various binary ops.
  if (auto bexpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
    if (bexpr.getKind() == AffineExprKind::Mul) {
      // For multiplies, RHS should always be constant of symbolic, and symbols
      // fail, so we cast to constant and give up if it doesn't work
      auto cexpr = bexpr.getRHS().dyn_cast<AffineConstantExpr>();
      if (!cexpr)
        return None;

      // Now compute the LHS via recursion
      auto out = computeStrideInfo(bexpr.getLHS(), args);
      if (!out)
        return None;

      // Multiply by the multiplier and return
      mulStrides(*out, cexpr.getValue());
      return out;
    }
    if (bexpr.getKind() == AffineExprKind::Add) {
      // For add, we compute both sides and add them (presuming they both return
      // valid outputs).
      auto out1 = computeStrideInfo(bexpr.getLHS(), args);
      if (!out1)
        return None;
      auto out2 = computeStrideInfo(bexpr.getRHS(), args);
      if (!out2)
        return None;
      addStrides(*out1, *out2);
      return out1;
    }
  }

  // Fail for all other cases.
  return None;
}

Optional<StrideInfo> computeStrideInfo(MemRefType memRefType, AffineMap map,
                                       ValueRange values) {
  // Verify the in/out dimensions make sense.
  assert(map.getNumResults() == memRefType.getRank());
  assert(map.getNumInputs() == values.size());

  // Get the memRef strides/offsets, and fail early is there is an isssue.
  int64_t memRefOffset;
  SmallVector<int64_t, 4> memRefStrides;
  if (failed(getStridesAndOffset(memRefType, memRefStrides, memRefOffset)))
    return None;

  // Fail if anything is dynamic.
  if (memRefOffset == MemRefType::kDynamicStrideOrOffset)
    return None;

  for (size_t i = 0; i < memRefStrides.size(); i++) {
    if (memRefStrides[i] == MemRefType::kDynamicStrideOrOffset)
      return None;
  }

  StrideInfo out;
  out.offset = memRefOffset;
  for (size_t i = 0; i < map.getNumResults(); i++) {
    // Collect the output for each dimension of the memRef.
    auto perDim = computeStrideInfo(map.getResult(i), values);

    // Fail if needed
    if (!perDim)
      return None;

    // Otherwise multiply by memRef stride and add in
    mulStrides(*perDim, memRefStrides[i]);
    addStrides(out, *perDim);
  }
  // Return the accumulated results
  return out;
}

Optional<StrideInfo> computeStrideInfo(AffineLoadOp op) {
  return computeStrideInfo(op.getMemRefType(), op.getAffineMap(),
                           op.getMapOperands());
}

Optional<StrideInfo> computeStrideInfo(AffineStoreOp op) {
  return computeStrideInfo(op.getMemRefType(), op.getAffineMap(),
                           op.getMapOperands());
}

Optional<StrideInfo> computeStrideInfo(pmlc::dialect::pxa::AffineReduceOp op) {
  return computeStrideInfo(op.out().getType().cast<MemRefType>(), op.map(),
                           op.idxs());
}

} // namespace mlir
