// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/stencil.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {

static constexpr StringLiteral kStencilAxisType = "stencil";

// A simple wrapper to provide an ordering to object vectors that
// we're going to be processing with std::next_permutation() --
// e.g. if we used pointers as comparison values, our order of
// iteration could vary run-to-run, creating non-determinism.
using OrderedValue = std::pair<unsigned, ValueStrideInfo>;

struct OrderedValueCmp {
  bool operator()(OrderedValue lhs, OrderedValue rhs) {
    return lhs.first < rhs.first;
  }
};

} // namespace

StencilBase::StencilBase(AffineParallelOp op,
                         ArrayRef<StencilIndexRequirement> requirements)
    : op(op), blockArgs(op.getIVs().begin(), op.getIVs().end()),
      requirements(requirements.begin(), requirements.end()),
      bestCost(std::numeric_limits<double>::infinity()),
      schedule(op->getAttrOfType<util::ScheduleAttr>(util::kScheduleAttrName)) {
}

void StencilBase::reportBestStencil(unsigned logLevel) {
  if (VLOG_IS_ON(logLevel)) {
    SmallVector<unsigned, 3> idxs = llvm::to_vector<3>(
        llvm::map_range(bestStencil.indexes,
                        [](BlockArgument idx) { return idx.getArgNumber(); }));
    std::stringstream ss;
    ss << "Stencil Selection Report:\n";
    ss << "    Best Perf: " << bestCost << "\n";
    ss << "    Best Tensor Permutation:\n";
    for (const ValueStrideInfo &vsi : bestStencil.values) {
      Value value = vsi.value;
      ss << "        " << debugString(value) << "\n";
    }
    ss << "    Best Index Permutation: " << idxs << '\n';
    ss << "    Best Tiling: " << bestTiling;
    IVLOG(logLevel, ss.str());
  }
}

std::vector<int64_t> StencilBase::generateTilings(int64_t idx, int64_t range) {
  std::pair<int64_t, int64_t> idxRangePair(idx, range);
  auto cached = tilingsCache.find(idxRangePair);
  if (cached != tilingsCache.end()) {
    return cached->second;
  }
  std::vector<int64_t> result = requirements[idx].tilingGenerator(range);
  tilingsCache.insert(std::make_pair(idxRangePair, result));
  return result;
}

int64_t StencilBase::getIdxRange(BlockArgument idx) {
  assert(getBlockArgsAsSet().count(idx) &&
         "getIdxRange only valid on indexes of current op");
  assert(idx.getArgNumber() < ranges.size());
  return ranges[idx.getArgNumber()];
}

StrideInfo StencilBase::getStrideInfo(Value value) {
  Optional<StrideInfo> maybeInfo =
      TypeSwitch<Operation *, Optional<StrideInfo>>(value.getDefiningOp())
          .Case<PxaLoadOp>([&](PxaLoadOp op) { return computeStrideInfo(op); })
          .Case<PxaReduceOp>(
              [&](PxaReduceOp op) { return computeStrideInfo(op); })
          .Default([](Operation *) { return None; });
  assert(maybeInfo.hasValue() && "StrideInfo must be computable");
  return *maybeInfo;
}

bool StencilIndexRequirement::check(ArrayRef<ValueStrideInfo> values,
                                    BlockArgument idx) const {
  assert(predicates.size() == values.size() &&
         "Each predicate entry must have one function per I/O op");
  for (unsigned i = 0; i < values.size(); i++) {
    const StrideInfo &info = values[i].strideInfo;
    int64_t stride = info.strides.lookup(idx);
    if (!predicates[i](stride))
      return false;
  }
  return true;
}

void StencilBase::bindIndexes(ArrayRef<ValueStrideInfo> values) {
  SetVector<BlockArgument> empty;
  recursiveBindIndex(empty, values);
}

void StencilBase::recursiveBindIndex(SetVector<BlockArgument> &boundIdxs,
                                     ArrayRef<ValueStrideInfo> values) {
  size_t currIdx = boundIdxs.size();
  if (currIdx == requirements.size()) {
    // This is a legal binding, go find a tiling for it
    SmallVector<int64_t, 8> currTileSize(requirements.size());
    recursiveTileIndex(StencilOption(values, boundIdxs.getArrayRef()),
                       currTileSize, 0);
  } else {
    Optional<util::AxisDim> axisDim;
    if (schedule)
      axisDim = schedule.getAxisResultDim(requirements[currIdx].idxName);

    for (BlockArgument blockArg : getBlockArgsAsSet()) {
      // Don't bind same index twice
      if (boundIdxs.contains(blockArg))
        continue;

      if (axisDim && blockArg.getArgNumber() != axisDim->dim)
        continue;

      // Verify the requirements for this index with each tensor are all met
      if (!requirements[currIdx].check(values, blockArg))
        continue;

      // If we made it to here, this index has appropriate semantics; bind it
      // and recurse
      boundIdxs.insert(blockArg);
      recursiveBindIndex(boundIdxs, values);
      boundIdxs.pop_back();
    }
  }
}

void StencilBase::recursiveTileIndex(const StencilOption &stencil,
                                     MutableArrayRef<int64_t> tileSizes,
                                     int64_t currIdx) {
  assert(tileSizes.size() == requirements.size());
  if (currIdx == requirements.size()) {
    IVLOG(3, "Considering Tile " << tileSizes);
    auto cost = getCost(stencil, tileSizes);
    IVLOG(3, "Tile cost = " << cost);
    if (cost < bestCost) {
      bestCost = cost;
      bestStencil = stencil;
      bestTiling.assign(tileSizes.begin(), tileSizes.end());
    }
  } else {
    assert(getBlockArgsAsSet().count(stencil.indexes[currIdx]) &&
           "BlockArg for current index must be valid");

    if (schedule) {
      if (Optional<util::AxisDim> axisDim =
              schedule.getAxisResultDim(requirements[currIdx].idxName)) {
        tileSizes[currIdx] = axisDim->axis.getRange();
        recursiveTileIndex(stencil, tileSizes, currIdx + 1);
        return;
      }
    }

    for (int64_t currIdxTileSize : generateTilings(
             currIdx, ranges[stencil.indexes[currIdx].getArgNumber()])) {
      tileSizes[currIdx] = currIdxTileSize;
      recursiveTileIndex(stencil, tileSizes, currIdx + 1);
    }
  }
}

void StencilBase::performStenciling() {
  // Initialization
  auto maybeRanges = op.getConstantRanges();
  if (!maybeRanges) {
    IVLOG(4, "Cannot Stencil: Requires constant ranges");
    return;
  }
  ranges = *maybeRanges;
  assert(ranges.size() == getBlockArgsAsSet().size());

  Optional<StencilCapture> maybeCapturedValues = capture();
  if (!maybeCapturedValues) {
    IVLOG(4, "Cannot Stencil: Operations fail to pattern-match.");
    return;
  }
  capturedValues = *maybeCapturedValues;

  // We wrap loads & stores with `OrderedValue` to make the order the
  // permutations are iterated through deterministic (the "sorted" order of the
  // IO ops is the order they were returned by `capture`) -- without this, the
  // sorted order would be however the pointers were ordered in memory.
  SmallVector<OrderedValue, 3> ordered;
  OrderedValueCmp cmp;
  for (Value value : capturedValues.stores) {
    ordered.push_back(std::make_pair(
        ordered.size(), ValueStrideInfo{value, getStrideInfo(value)}));
  }
  size_t firstLoadIdx = ordered.size();
  for (Value value : capturedValues.loads) {
    ordered.push_back(std::make_pair(
        ordered.size(), ValueStrideInfo{value, getStrideInfo(value)}));
  }
  auto itLastStoreFirstLoad = ordered.begin() + firstLoadIdx;
  std::sort(ordered.begin(), itLastStoreFirstLoad, cmp);
  do { // Each store tensor permutation
    std::sort(itLastStoreFirstLoad, ordered.end(), cmp);
    do { // Each load tensor permutation
      SmallVector<ValueStrideInfo, 3> values =
          llvm::to_vector<3>(llvm::map_range(
              ordered, [](OrderedValue value) { return value.second; }));
      bindIndexes(values);
    } while (std::next_permutation(itLastStoreFirstLoad, ordered.end(), cmp));
  } while (std::next_permutation(ordered.begin(), itLastStoreFirstLoad, cmp));

  if (bestCost < std::numeric_limits<double>::infinity()) {
    reportBestStencil(2);
    transform(bestStencil, bestTiling);
    if (schedule) {
      DenseSet<StringRef> usedIdxs;
      for (const StencilIndexRequirement &req : requirements) {
        usedIdxs.insert(req.idxName);
      }

      util::ScheduleAttr newSchedule = schedule.removeAxes(usedIdxs);
      if (newSchedule)
        op->setAttr(util::kScheduleAttrName, newSchedule);
      else
        op->removeAttr(util::kScheduleAttrName);
    }
  } else {
    IVLOG(3, "No legal tiling found to stencil");
  }
}

AffineMap makeTileMap(MLIRContext *context, AffineMap map, ValueRange operands,
                      ArrayRef<BlockArgument> idxs) {
  SmallVector<AffineExpr, 8> exprs;
  for (Value value : operands) {
    bool found = false;
    for (size_t i = 0; i < idxs.size(); i++) {
      if (value == idxs[i]) {
        exprs.push_back(getAffineDimExpr(i, context));
        found = true;
      }
    }
    if (!found) {
      exprs.push_back(getAffineConstantExpr(0, context));
    }
  }
  auto toIdxs = AffineMap::get(idxs.size(), 0, exprs, context);
  return map.compose(toIdxs);
}

} // namespace pmlc::dialect::pxa
