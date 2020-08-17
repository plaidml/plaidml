// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/analysis/strides.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Support/DebugStringHelper.h"
#include "pmlc/dialect/pxa/analysis/affine_expr.h"
#include "pmlc/util/logging.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {

namespace pxa = pmlc::dialect::pxa;
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

// Generally useful helper function
int64_t getIVStep(BlockArgument arg) {
  // Check the kind of loop we are part of, and dispatch.
  Operation *baseOp = arg.getOwner()->getParentOp();

  size_t idx = arg.getArgNumber();
  if (auto op = dyn_cast<AffineParallelOp>(baseOp)) {
    auto stepAttr = op.steps().getValue()[idx];
    return stepAttr.cast<IntegerAttr>().getInt();
  }
  if (auto op = dyn_cast<AffineForOp>(baseOp)) {
    return op.getStep();
  }
  llvm_unreachable("Get IV Step on non-IV");
}

StrideRange::StrideRange(BlockArgument arg)
    : valid(false), minVal(0), maxVal(0), stride(0) {
  if (auto ap =
          mlir::dyn_cast<AffineParallelOp>(arg.getOwner()->getParentOp())) {
    auto range_expr = ap.getRangesValueMap().getResult(arg.getArgNumber());
    auto range_cst = range_expr.dyn_cast<AffineConstantExpr>();
    if (!range_cst) {
      return;
    }
    int64_t range = range_cst.getValue();
    if (range < 1) {
      return;
    }
    int64_t step = ap.steps()[arg.getArgNumber()].cast<IntegerAttr>().getInt();
    if (step <= 0) {
      return;
    }
    stride = 1;
    minVal = 0;
    // This is a correction to deal with the fact that strides are measured
    // relative to loop iterations not indexes.
    maxVal = (range - 1) / step;
    valid = true;
    if (minVal == maxVal) {
      stride = 0;
    }
  }
}

StrideRange &StrideRange::operator*=(int64_t factor) {
  minVal *= factor;
  maxVal *= factor;
  stride *= factor;
  if (factor < 0) {
    std::swap(minVal, maxVal);
  }
  return *this;
}

StrideRange &StrideRange::operator+=(const StrideRange &rhs) {
  valid = valid && rhs.valid;
  minVal += rhs.minVal;
  maxVal += rhs.maxVal;
  stride = std::gcd(stride, rhs.stride);
  return *this;
}

void StrideRange::unionEquals(const StrideRange &rhs) {
  valid = valid && rhs.valid;
  minVal = std::min(minVal, rhs.minVal);
  maxVal = std::max(maxVal, rhs.maxVal);
  stride = std::gcd(stride, rhs.stride);
}

// Multiply the offset and all strides by a constant.
StrideInfo &StrideInfo::operator*=(int64_t factor) {
  offset *= factor;
  if (factor == 0) {
    strides.clear();
  } else {
    for (auto &kvp : strides) {
      kvp.second *= factor;
    }
  }
  return *this;
}

// StrideInfo addition operation.
StrideInfo &StrideInfo::operator+=(const StrideInfo &rhs) {
  offset += rhs.offset;
  for (const auto &kvp : rhs.strides) {
    strides[kvp.first] += kvp.second;
  }
  // Remove entries with 0 for stride
  for (auto &kvp : llvm::make_early_inc_range(strides)) {
    if (kvp.second == 0) {
      // DenseMap never resizes during erase, so iterators stay valid.
      strides.erase(kvp.first);
    }
  }
  return *this;
}

static bool isBlockAncestor(Block *x, Block *y) {
  while (x != y) {
    Operation *parentOp = y->getParentOp();
    if (!parentOp) {
      return false;
    }
    y = parentOp->getBlock();
    if (!y) {
      return false;
    }
  }
  return true;
}

StrideInfo StrideInfo::outer(Block *block) {
  StrideInfo ret;
  ret.offset = offset;
  for (const auto &kvp : strides) {
    if (isBlockAncestor(kvp.first.getOwner(), block)) {
      ret.strides.insert(kvp);
    }
  }
  return ret;
}

StrideInfo StrideInfo::inner(Block *block) {
  StrideInfo ret;
  ret.offset = 0;
  for (const auto &kvp : strides) {
    if (!isBlockAncestor(kvp.first.getOwner(), block)) {
      ret.strides.insert(kvp);
    }
  }
  return ret;
}

StrideRange StrideInfo::range() const {
  StrideRange ret(offset);
  for (const auto &kvp : strides) {
    ret += StrideRange(kvp.first) * kvp.second;
  }
  return ret;
}

AffineValueExpr StrideInfo::toValueExpr(MLIRContext *ctx) const {
  auto tot = AffineValueExpr(ctx, offset);
  for (const auto &kvp : strides) {
    Operation *baseOp = kvp.first.getOwner()->getParentOp();
    AffineValueExpr idx(kvp.first);
    if (auto op = dyn_cast<AffineParallelOp>(baseOp)) {
      idx = idx - AffineValueExpr(op.getLowerBoundsValueMap(),
                                  kvp.first.getArgNumber());
    } else if (auto op = dyn_cast<AffineForOp>(baseOp)) {
      auto map = op.getLowerBoundMap();
      idx = idx - AffineValueExpr(map.getResult(0), op.getLowerBoundOperands());
    } else {
      llvm_unreachable("Invalid op type in toValueMap");
    }
    int64_t step = getIVStep(kvp.first);
    assert(kvp.second % step == 0 && "Stride not divisible by step");
    tot = tot + idx * (kvp.second / step);
  }
  return tot;
}

void StrideInfo::print(raw_ostream &os, Block *relative) const {
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

AffineValueMap convertToValueMap(MLIRContext *ctx, ArrayRef<StrideInfo> dims) {
  SmallVector<AffineValueExpr, 4> exprs;
  for (const auto &si : dims) {
    exprs.push_back(si.toValueExpr(ctx));
  }
  return jointValueMap(ctx, exprs);
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
  if (auto op = dyn_cast_or_null<AffineApplyOp>(expr.getDefiningOp()))
    return computeStrideInfo(op.getAffineMap().getResult(0),
                             op.getMapOperands());

  IVLOG(1, "Failed stride info: op = " << mlir::debugString(expr));

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

Optional<llvm::SmallVector<StrideInfo, 4>> computeStrideInfo(AffineMap map,
                                                             ValueRange args) {
  llvm::SmallVector<StrideInfo, 4> results;
  for (auto expr : map.getResults()) {
    auto dimStride = computeStrideInfo(expr, args);
    if (!dimStride) {
      return None;
    }
    results.push_back(*dimStride);
  }
  return results;
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

Optional<StrideInfo> computeStrideInfo(pxa::PxaLoadOp op) {
  return computeStrideInfo(op.getMemRefType(), op.getAffineMap(),
                           op.getMapOperands());
}

Optional<StrideInfo> computeStrideInfo(pxa::PxaReduceOp op) {
  return computeStrideInfo(op.getMemRefType(), op.getAffineMap(),
                           op.getMapOperands());
}

Optional<StrideInfo> computeStrideInfo(pxa::PxaVectorLoadOp op) {
  return computeStrideInfo(op.getMemRefType(), op.getAffineMap(),
                           op.getMapOperands());
}

Optional<StrideInfo> computeStrideInfo(pxa::PxaVectorReduceOp op) {
  return computeStrideInfo(op.getMemRefType(), op.getAffineMap(),
                           op.getMapOperands());
}

Optional<RelativeAccessPattern> computeRelativeAccess(Block *block,
                                                      Operation *op) {
  ArrayRef<int64_t> vecSize = {};
  Optional<llvm::SmallVector<StrideInfo, 4>> maybeStrides;
  TypeSwitch<Operation *>(op)
      .Case<pxa::PxaLoadOp>([&](auto op) {
        maybeStrides =
            computeStrideInfo(op.getAffineMap(), op.getMapOperands());
      })
      .Case<pxa::PxaReduceOp>([&](auto op) {
        maybeStrides =
            computeStrideInfo(op.getAffineMap(), op.getMapOperands());
      })
      .Case<pxa::PxaVectorLoadOp>([&](auto op) {
        maybeStrides =
            computeStrideInfo(op.getAffineMap(), op.getMapOperands());
        vecSize = op.getVectorType().getShape();
      })
      .Case<pxa::PxaVectorReduceOp>([&](auto op) {
        maybeStrides =
            computeStrideInfo(op.getAffineMap(), op.getMapOperands());
        vecSize = op.getVectorType().getShape();
      });
  if (!maybeStrides) {
    return llvm::None;
  }
  auto &strides = *maybeStrides;
  RelativeAccessPattern ret;
  for (size_t i = 0; i < strides.size(); i++) {
    ret.outer.push_back(strides[i].outer(block));
    auto inner = strides[i].inner(block);
    ret.inner.push_back(inner);
    StrideRange range = inner.range();
    if (i + vecSize.size() >= strides.size()) {
      int64_t vecVal = vecSize[i + vecSize.size() - strides.size()];
      if (vecVal > 1) {
        StrideRange vecRange(0, vecVal - 1, 1);
        range += vecRange;
      }
    }
    if (!range.valid || range.minVal != 0) {
      return llvm::None;
    }
    ret.innerCount.push_back(range.count());
    int64_t stride = range.stride;
    if (stride == 0) {
      stride = 1;
    }
    ret.innerStride.push_back(stride);
  }
  return ret;
}

StrideArray::StrideArray(unsigned numDims, int64_t offset)
    : offset(offset), strides(numDims) {}

StrideArray &StrideArray::operator*=(int64_t factor) {
  offset *= factor;
  for (auto &dim : strides)
    dim *= factor;
  return *this;
}

StrideArray &StrideArray::operator+=(const StrideArray &rhs) {
  assert(strides.size() == rhs.strides.size() && "strides sizes much match");
  offset += rhs.offset;
  for (unsigned i = 0, e = strides.size(); i < e; ++i) {
    strides[i] += rhs.strides[i];
  }
  return *this;
}

void StrideArray::print(raw_ostream &os) {
  os << offset << ":[";
  for (auto item : llvm::enumerate(strides)) {
    if (item.index())
      os << ", ";
    os << item.value();
  }
  os << ']';
}

Optional<StrideArray> computeStrideArray(AffineMap map) {
  std::vector<SmallVector<int64_t, 8>> flat;
  if (failed(getFlattenedAffineExprs(map, &flat, nullptr)))
    return llvm::None;

  StrideArray ret(map.getNumDims(), flat.front().back());
  for (unsigned i = 0, e = map.getNumDims(); i < e; i++) {
    ret.strides[i] = flat.front()[i];
  }

  return ret;
}

} // namespace mlir
