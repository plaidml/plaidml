// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/analysis/strides.h"

#include <map>
#include <string>

#include "llvm/Support/FormatVariadic.h"

namespace mlir {

const char *kBlockAndArgFormat = "^bb{0}:%arg{1}";

static std::string getUniqueName(Block *ref, BlockArgument arg) {
  unsigned reverseDepth = 0;
  while (arg.getOwner() != ref) {
    ref = ref->getParentOp()->getBlock();
    reverseDepth++;
  }
  return llvm::formatv(kBlockAndArgFormat, reverseDepth, arg.getArgNumber())
      .str();
}

// Multiply the offset and all strides by a constant.
StrideInfo &StrideInfo::operator*=(int64_t factor) {
  offset *= factor;
  for (auto &kvp : strides) {
    kvp.second *= factor;
  }
  return *this;
}

// StrideInfo addition operation.
StrideInfo &StrideInfo::operator+=(const StrideInfo &rhs) {
  offset += rhs.offset;
  for (const auto &kvp : rhs.strides) {
    strides[kvp.first] += kvp.second;
  }
  return *this;
}

void StrideInfo::print(raw_ostream &os, Block *relative) {
  std::map<std::string, unsigned> ordered;
  std::map<Block *, unsigned> blockIds;
  for (auto kvp : strides) {
    if (relative) {
      ordered.emplace(getUniqueName(relative, kvp.first), kvp.second);
    } else {
      auto itNew = blockIds.emplace(kvp.first.getOwner(), blockIds.size());
      ordered.emplace(llvm::formatv(kBlockAndArgFormat, itNew.first->second,
                                    kvp.first.getArgNumber()),
                      kvp.second);
    }
  }
  os << offset << ":[";
  for (auto item : llvm::enumerate(ordered)) {
    if (item.index())
      os << ", ";
    os << item.value().first << "=" << item.value().second;
  }
  os << ']';
}

static Optional<StrideInfo> computeStrideInfo(AffineParallelOp op,
                                              BlockArgument arg) {
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
  if (auto cexpr = expr.dyn_cast<AffineConstantExpr>())
    return StrideInfo(cexpr.getValue());

  // If we are a dim, it's just a Value.
  if (auto dexpr = expr.dyn_cast<AffineDimExpr>())
    return computeStrideInfo(args[dexpr.getPosition()]);

  // Check the various binary ops.
  if (auto bexpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
    if (bexpr.getKind() == AffineExprKind::Mul) {
      // For multiplies, RHS should always be constant of symbolic, and symbols
      // fail, so we cast to constant and give up if it doesn't work
      auto rhs = bexpr.getRHS().dyn_cast<AffineConstantExpr>();
      if (!rhs)
        return None;

      // Now compute the LHS via recursion
      auto lhs = computeStrideInfo(bexpr.getLHS(), args);
      if (!lhs)
        return None;

      // Multiply by the multiplier and return
      *lhs *= rhs.getValue();
      return lhs;
    }

    if (bexpr.getKind() == AffineExprKind::Add) {
      // For add, we compute both sides and add them (presuming they both return
      // valid outputs).
      auto lhs = computeStrideInfo(bexpr.getLHS(), args);
      if (!lhs)
        return None;
      auto rhs = computeStrideInfo(bexpr.getRHS(), args);
      if (!rhs)
        return None;
      *lhs += *rhs;
      return lhs;
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

  // Get the memRef strides/offsets, and fail early if there is an isssue.
  int64_t memRefOffset;
  SmallVector<int64_t, 4> memRefStrides;
  if (failed(getStridesAndOffset(memRefType, memRefStrides, memRefOffset)))
    return None;

  // Fail if anything is dynamic.
  if (ShapedType::isDynamicStrideOrOffset(memRefOffset))
    return None;

  for (size_t i = 0; i < memRefStrides.size(); i++) {
    if (ShapedType::isDynamicStrideOrOffset(memRefStrides[i]))
      return None;
  }

  StrideInfo out(memRefOffset);
  for (size_t i = 0; i < map.getNumResults(); i++) {
    // Collect the output for each dimension of the memRef.
    auto perDim = computeStrideInfo(map.getResult(i), values);

    // Fail if needed
    if (!perDim)
      return None;

    // Otherwise multiply by memRef stride and add in
    *perDim *= memRefStrides[i];
    out += *perDim;
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
