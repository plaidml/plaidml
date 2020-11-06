// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/analysis/strides.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/IR/Function.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "pmlc/dialect/pxa/analysis/affine_expr.h"
#include "pmlc/util/bilp/ilp_solver.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

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
    auto steps = op.getSteps();
    return steps[idx];
  }
  if (auto op = dyn_cast<AffineForOp>(baseOp)) {
    return op.getStep();
  }
  llvm_unreachable("Get IV Step on non-IV");
}

static Optional<StrideInfo> flatten(MemRefType memRefType,
                                    ArrayRef<StrideInfo> dimensional) {
  assert(memRefType.getRank() == static_cast<int64_t>(dimensional.size()) &&
         "memRef and dimensional rank mismatch");
  // Get the memRef strides/offsets, and fail early if there is an issue.
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(memRefType, strides, offset)))
    return None;

  // Fail if anything is dynamic.
  if (ShapedType::isDynamicStrideOrOffset(offset) ||
      llvm::any_of(strides, ShapedType::isDynamicStrideOrOffset))
    return None;

  StrideInfo flat{offset};
  for (size_t i = 0; i < strides.size(); i++) {
    flat += dimensional[i] * strides[i];
  }

  return flat;
}

StrideRange::StrideRange(BlockArgument arg)
    : valid(false), minVal(0), maxVal(0), stride(0) {
  if (auto ap = dyn_cast<AffineParallelOp>(arg.getOwner()->getParentOp())) {
    auto rangeExpr = ap.getRangesValueMap().getResult(arg.getArgNumber());
    auto rangeConstantExpr = rangeExpr.dyn_cast<AffineConstantExpr>();
    if (!rangeConstantExpr) {
      return;
    }
    int64_t range = rangeConstantExpr.getValue();
    if (range < 1) {
      return;
    }
    auto steps = ap.getSteps();
    int64_t step = steps[arg.getArgNumber()];
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

static BoundaryRegion getBoundaryRegion(Block *x, Block *y) {
  while (x != y) {
    Operation *parentOp = y->getParentOp();
    if (!parentOp) {
      return BoundaryRegion::Interior;
    }
    y = parentOp->getBlock();
    if (!y) {
      return BoundaryRegion::Interior;
    }
  }
  return BoundaryRegion::Exterior;
}

StrideInfo StrideInfo::outer(BlockArgumentBoundaryFn fn) {
  StrideInfo ret;
  ret.offset = offset;
  for (const auto &kvp : strides) {
    if (fn(kvp.first) == BoundaryRegion::Exterior) {
      ret.strides.insert(kvp);
    }
  }
  return ret;
}

StrideInfo StrideInfo::inner(BlockArgumentBoundaryFn fn) {
  StrideInfo ret;
  ret.offset = 0;
  for (const auto &kvp : strides) {
    if (fn(kvp.first) == BoundaryRegion::Interior) {
      ret.strides.insert(kvp);
    }
  }
  return ret;
}

StrideInfo StrideInfo::outer(Block *block) {
  return outer([block](BlockArgument arg) {
    return getBoundaryRegion(arg.getOwner(), block);
  });
}

StrideInfo StrideInfo::inner(Block *block) {
  return inner([block](BlockArgument arg) {
    return getBoundaryRegion(arg.getOwner(), block);
  });
}

StrideRange StrideInfo::range() const {
  StrideRange ret(offset);
  for (const auto &kvp : strides) {
    ret += StrideRange(kvp.first) * kvp.second;
  }
  return ret;
}

AffineValueExpr StrideInfo::toValueExpr(MLIRContext *ctx) const {
  typedef std::pair<unsigned, unsigned> nestedArgNumber;
  std::map<nestedArgNumber, mlir::BlockArgument> ordered;

  for (auto kvp : strides) {
    unsigned loopDepth = 0;
    auto parent = kvp.first.getOwner()->getParentOp();
    while (!dyn_cast<FuncOp>(parent)) {
      loopDepth++;
      parent = parent->getParentOp();
    }

    ordered.emplace(nestedArgNumber(loopDepth, kvp.first.getArgNumber()),
                    kvp.first);
  }

  auto tot = AffineValueExpr(ctx, offset);
  for (auto item : llvm::enumerate(ordered)) {
    auto blockArg = item.value().second;
    Operation *baseOp = blockArg.getOwner()->getParentOp();
    AffineValueExpr idx(blockArg);
    if (auto op = dyn_cast<AffineParallelOp>(baseOp)) {
      idx = idx - AffineValueExpr(op.getLowerBoundsValueMap(),
                                  blockArg.getArgNumber());
    } else if (auto op = dyn_cast<AffineForOp>(baseOp)) {
      auto map = op.getLowerBoundMap();
      idx = idx - AffineValueExpr(map.getResult(0), op.getLowerBoundOperands());
    } else {
      llvm_unreachable("Invalid op type in toValueMap");
    }
    int64_t step = getIVStep(blockArg);
    auto si = strides.find(blockArg);
    assert(si != strides.end());
    assert(si->second % step == 0 && "Stride not divisible by step");
    tot = tot + idx * (si->second / step);
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
  os << offset;
  if (ordered.size()) {
    os << ":[";
    for (auto item : llvm::enumerate(ordered)) {
      if (item.index())
        os << ", ";
      os << item.value().first << "=" << item.value().second;
    }
    os << ']';
  }
}

std::ostream &operator<<(std::ostream &os, const StrideInfo &x) {
  os << debugString(x);
  return os;
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
  auto steps = op.getSteps();
  out->strides[arg] += steps[idx];
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

  IVLOG(1, "Failed stride info: op = " << debugString(expr));

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

Optional<SmallVector<StrideInfo, 4>> computeStrideInfo(AffineMap map,
                                                       ValueRange args) {
  SmallVector<StrideInfo, 4> results;
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

  // Get the memRef strides/offsets, and fail early if there is an issue.
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

Optional<StrideInfo> computeStrideInfo(PxaLoadOp op) {
  return computeStrideInfo(op.getMemRefType(), op.getAffineMap(),
                           op.getMapOperands());
}

Optional<StrideInfo> computeStrideInfo(PxaReduceOp op) {
  return computeStrideInfo(op.getMemRefType(), op.getAffineMap(),
                           op.getMapOperands());
}

Optional<StrideInfo> computeStrideInfo(PxaVectorLoadOp op) {
  return computeStrideInfo(op.getMemRefType(), op.getAffineMap(),
                           op.getMapOperands());
}

Optional<StrideInfo> computeStrideInfo(PxaVectorReduceOp op) {
  return computeStrideInfo(op.getMemRefType(), op.getAffineMap(),
                           op.getMapOperands());
}

Optional<RelativeAccessPattern>
computeRelativeAccess(Operation *op, BlockArgumentBoundaryFn fn) {
  ArrayRef<int64_t> vecSize;
  Value memref;
  using MaybeStrides = Optional<SmallVector<StrideInfo, 4>>;
  auto maybeStrides =
      TypeSwitch<Operation *, MaybeStrides>(op)
          .Case<PxaLoadOp>([&](auto op) {
            memref = op.memref();
            return computeStrideInfo(op.getAffineMap(), op.getMapOperands());
          })
          .Case<PxaReduceOp>([&](auto op) {
            memref = op.memref();
            return computeStrideInfo(op.getAffineMap(), op.getMapOperands());
          })
          .Case<PxaVectorLoadOp>([&](auto op) {
            memref = op.memref();
            vecSize = op.getVectorType().getShape();
            return computeStrideInfo(op.getAffineMap(), op.getMapOperands());
          })
          .Case<PxaVectorReduceOp>([&](auto op) {
            memref = op.memref();
            vecSize = op.getVectorType().getShape();
            return computeStrideInfo(op.getAffineMap(), op.getMapOperands());
          })
          .Default([](auto op) { return None; });
  if (!maybeStrides) {
    return None;
  }
  auto &strides = *maybeStrides;
  RelativeAccessPattern ret(memref);
  for (size_t i = 0; i < strides.size(); i++) {
    auto outer = strides[i].outer(fn);
    ret.outer.push_back(outer);
    auto inner = strides[i].inner(fn);
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
      return None;
    }
    ret.innerRanges.push_back(range);
    ret.innerCount.push_back(range.count());
  }
  return ret;
}

Optional<RelativeAccessPattern> computeRelativeAccess(Operation *op,
                                                      Block *block) {
  return computeRelativeAccess(op, [block](BlockArgument arg) {
    return getBoundaryRegion(arg.getOwner(), block);
  });
}

MemRefType RelativeAccessPattern::getMemRefType() const {
  return memRef.getType().cast<MemRefType>();
}

SmallVector<int64_t, 4> RelativeAccessPattern::innerStride() const {
  SmallVector<int64_t, 4> ret;
  for (const StrideRange &range : innerRanges) {
    ret.push_back(range.stride ? range.stride : 1);
  }
  return ret;
}

int64_t RelativeAccessPattern::totalInnerCount() const {
  int64_t ret = 1;
  for (auto range : innerRanges) {
    ret *= range.count();
  }
  return ret;
}

int64_t RelativeAccessPattern::totalInnerBytes() const {
  auto eltSize = llvm::divideCeil(getMemRefType().getElementTypeBitWidth(), 8);
  return totalInnerCount() * eltSize;
}

Optional<StrideInfo> RelativeAccessPattern::flatOuter() const {
  return flatten(getMemRefType(), outer);
}

Optional<StrideInfo> RelativeAccessPattern::flatInner() const {
  return flatten(getMemRefType(), inner);
}

LogicalResult
RelativeAccessPattern::unionMerge(const RelativeAccessPattern &rhs) {
  if (innerRanges.size() != rhs.innerRanges.size()) {
    return failure();
  }

  for (unsigned i = 0; i < outer.size(); i++) {
    if (outer[i] != rhs.outer[i]) {
      return failure();
    }
  }

  inner.clear();
  innerCount.clear();

  for (unsigned i = 0; i < innerRanges.size(); i++) {
    innerRanges[i].unionEquals(rhs.innerRanges[i]);
    innerCount.push_back(innerRanges[i].count());
  }

  return success();
}

// Use ILP to find out if two different iterations of the outer indexes (in
// allOuter) ever alias.  To do this, we make a system with two copies of all
// the variable (outer + inner indexes) called _a and _b.  Then we constrain the
// total effect of _a and the total effect of _b to be the same (i.e. both
// index sets access the same memory location).  We also contrain the indexes to
// their appropriate ranges.  Then we see if we can ever get the _a and _b
// version of any of the outer indexes to differ at all (by minimimizing oi_a -
// oi_b).  If it's possible for them the differ, then (since _a + _b are
// symmetric), the minimum of the difference will be < 0.
bool RelativeAccessPattern::outerAlias(DenseSet<BlockArgument> allOuter) const {
  using Poly = util::math::Polynomial<util::math::Rational>;
  using RangeCons = util::math::RangeConstraint;
  util::bilp::ILPSolver solver;
  // We track each index as we add it, and make a string version since the
  // Polynomial logic uses string names for variables.
  DenseMap<BlockArgument, std::string> baToStr;
  // The collection of constraints, this ends up including:
  // 1) Range constraints for two copies (a + b).
  // 2) Contraints requiring all dimensions of the access to be the same for
  //    both the a access and the b access.
  std::vector<RangeCons> constraints;
  // The collection of things to minimize.  Here it's the differences of _a and
  // _b for all outer indexes
  std::vector<Poly> toMin;
  // A lambda to convert a block arg to x<i>_a - x<i>_b for some unique i.  If
  // we haven't seen the block arg before, we add it to the map, along with
  // range constraints for the _a and _b versions.
  auto toDiff = [&](BlockArgument arg) {
    std::string &str = baToStr[arg];
    if (str.empty()) {
      str = llvm::formatv("x{0}", baToStr.size());
      StrideRange range(arg);
      constraints.emplace_back(str + "_a", range.maxVal + 1);
      constraints.emplace_back(str + "_b", range.maxVal + 1);
    }
    return Poly(str + "_a") - Poly(str + "_b");
  };
  // Add entries for all the outer indexes.
  for (auto &arg : allOuter) {
    auto diff = toDiff(arg);
    toMin.emplace_back(diff);
  }
  // Go over each dimension of the access.
  for (size_t i = 0; i < outer.size(); i++) {
    // Compute the difference between the a + b versions of the access for this
    // dimension of the access.
    Poly totDiff;
    for (const auto &kvp : outer[i].strides) {
      auto diff = toDiff(kvp.first);
      totDiff += diff * kvp.second;
    }
    for (const auto &kvp : inner[i].strides) {
      auto diff = toDiff(kvp.first);
      totDiff += diff * kvp.second;
    }
    // Constrain this diff to be the range [0, 1) in integers, i.e. constrain it
    // to be exactly 0.
    constraints.emplace_back(totDiff, 1);
  }
  IVLOG(3, "Doing a batch solve!");
  IVLOG(3, "Constraints: " << constraints);
  IVLOG(3, "toMin: " << toMin);
  // Do the actual ILP solve.
  auto res = solver.batch_solve(constraints, toMin);
  // If any of the results have a minimum that is not exactly 0, it means the _a
  // and _b version of that index can have two differnt values while still
  // accessing the same point in the tensor.  AKA we have hit an outer alias.
  for (const auto &kvp : res) {
    IVLOG(3, "obj_val = " << kvp.second.obj_val);
    if (kvp.second.obj_val < 0) {
      return true;
    }
  }
  // Otherwise, all is well.
  return false;
}

bool hasPerfectAliasing(const RelativeAccessPattern &aRap,
                        RelativeAccessPattern bRap,
                        const DenseMap<BlockArgument, BlockArgument> &bToA) {
  DenseSet<BlockArgument> allOuter;
  for (const auto &kvp : bToA) {
    allOuter.insert(kvp.second);
  }
  if (aRap.outerAlias(allOuter)) {
    IVLOG(3, "outerAlias");
    return false;
  }
  for (auto &si : bRap.outer) {
    StrideInfo translated(si.offset);
    for (const auto &kvp : si.strides) {
      translated.strides[bToA.find(kvp.first)->second] = kvp.second;
    }
    si = translated;
  }
  if (aRap.outer.size() != bRap.outer.size()) {
    IVLOG(3, "size mismatch: " << aRap.outer.size()
                               << " != " << bRap.outer.size());
    return false;
  }
  for (size_t i = 0; i < aRap.outer.size(); i++) {
    const StrideInfo &aOuter = aRap.outer[i];
    const StrideInfo &bOuter = bRap.outer[i];
    const int64_t aInnerCount = aRap.innerCount[i];
    const int64_t bInnerCount = bRap.innerCount[i];
    if (aOuter != bOuter) {
      IVLOG(3, "aOuter != bOuter");
      return false;
    }
    if (aInnerCount != bInnerCount) {
      IVLOG(3, "aInnerCount != bInnerCount");
      return false;
    }
  }
  return true;
}

double computeCacheMiss(double cacheElems,
                        SmallVector<int64_t, 4> tileDimensions,
                        SmallVector<int64_t, 4> tensorStrides) {
  // Start with one cache line
  double cacheLines = 1.0;
  // Current accumulated maximum value
  int64_t maxVal = 0;
  // For each dimension (in sorted order)
  assert(tileDimensions.size() == tensorStrides.size());
  for (size_t i = tileDimensions.size(); i > 0; i--) {
    // Compute gap per step
    int64_t gap = std::abs(tensorStrides[i - 1]) - maxVal;
    // Multiply current cache hits by size
    cacheLines *= static_cast<double>(tileDimensions[i - 1]);
    // Compute probability that cache line is shared across gap
    double probShared = 0.0; // Assume it's never shared
    if (cacheElems != 0.0 && gap < cacheElems) {
      probShared = 1.0 - (gap / cacheElems);
    }
    // Subtract shared elements
    cacheLines -= probShared * static_cast<double>(tileDimensions[i - 1] - 1);
    // Update maxVal
    maxVal += std::abs(tensorStrides[i - 1]) * (tileDimensions[i - 1] - 1);
  }
  return cacheLines;
}

} // namespace pmlc::dialect::pxa
