// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/transforms/padding.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/passes.h"

namespace pmlc::dialect::tile {

using mlir::AffineMap;

struct BoundRange {
  int64_t min;
  int64_t max;

  explicit BoundRange(int64_t val) : min(val), max(val) {}

  BoundRange(int64_t min, int64_t max) : min(min), max(max) {}

  BoundRange operator+(const BoundRange &rhs) const {
    return {min + rhs.min, max + rhs.max};
  }

  BoundRange operator*(const BoundRange &rhs) const {
    return {std::min(std::min(min * rhs.min, max * rhs.min),
                     std::min(min * rhs.max, max * rhs.max)),
            std::max(std::max(min * rhs.min, max * rhs.min),
                     std::max(min * rhs.max, max * rhs.max))};
  }
};

class ComputeRangeVisitor
    : public mlir::AffineExprVisitor<ComputeRangeVisitor, BoundRange> {
public:
  explicit ComputeRangeVisitor(std::vector<BoundRange> dimRanges)
      : dimRanges(std::move(dimRanges)) {}

  BoundRange visitMulExpr(mlir::AffineBinaryOpExpr expr) {
    return visit(expr.getLHS()) * visit(expr.getRHS());
  }

  BoundRange visitAddExpr(mlir::AffineBinaryOpExpr expr) {
    return visit(expr.getLHS()) + visit(expr.getRHS());
  }

  BoundRange visitDimExpr(mlir::AffineDimExpr expr) {
    return dimRanges[expr.getPosition()];
  }

  BoundRange visitSymbolExpr(mlir::AffineSymbolExpr expr) {
    llvm_unreachable("Invalid symbol in ComputeRangeVisitor");
  }

  BoundRange visitConstantExpr(mlir::AffineConstantExpr expr) {
    return BoundRange(expr.getValue());
  }

  BoundRange visitModExpr(mlir::AffineBinaryOpExpr expr) {
    llvm_unreachable("Invalid mod in ComputeRangeVisitor");
  }

  BoundRange visitCeilDivExpr(mlir::AffineBinaryOpExpr expr) {
    llvm_unreachable("Invalid ceil in ComputeRangeVisitor");
  }

  BoundRange visitFloorDivExpr(mlir::AffineBinaryOpExpr expr) {
    llvm_unreachable("Invalid floor in ComputeRangeVisitor");
  }

private:
  std::vector<BoundRange> dimRanges;
};

llvm::SmallVector<BoundRange, 4>
computePaddingBounds(AffineMap access, AffineMap lower, AffineMap upper) {
  // We only handle the 'easy' cases.
  assert(access.getNumSymbols() == 0);
  assert(lower.getNumSymbols() == 0);
  assert(upper.getNumSymbols() == 0);
  assert(lower.getNumDims() == 0);
  assert(upper.getNumDims() == 0);
  unsigned idxs = access.getNumDims();
  unsigned size = access.getNumResults();
  assert(lower.getNumResults() == idxs);
  assert(upper.getNumResults() == idxs);
  std::vector<BoundRange> dimRanges;
  for (unsigned i = 0; i < idxs; i++) {
    int64_t lowerBoundInt =
        lower.getResult(i).cast<mlir::AffineConstantExpr>().getValue();
    int64_t upperBoundInt =
        upper.getResult(i).cast<mlir::AffineConstantExpr>().getValue();
    dimRanges.emplace_back(lowerBoundInt, upperBoundInt);
  }
  ComputeRangeVisitor visitor(std::move(dimRanges));
  llvm::SmallVector<BoundRange, 4> outRanges;
  for (unsigned i = 0; i < size; i++) {
    outRanges.push_back(visitor.visit(access.getResult(i)));
  }
  return outRanges;
}

static bool validForPadding(AggregationKind agg, CombinationKind comb) {
  if (agg == AggregationKind::assign) {
    return false;
  }
  if (comb == CombinationKind::none) {
    return true;
  }
  if (comb == CombinationKind::mul && agg == AggregationKind::add) {
    return true;
  }
  return false;
}

void PadPass::runOnFunction() {
  auto func = getFunction();
  llvm::DenseMap<Value, llvm::DenseMap<AggregationKind, PaddingInfo>> toPad;

  func.walk([&](ContractionOp op) {
    // Skip some cases where the padding pass can't operate.
    if (op.getNumSymbols()) {
      op.emitRemark("padding cannot be run on symbolic contractions");
      return;
    }

    if (!op.lowerBounds() || !op.upperBounds()) {
      op.emitRemark("contraction bounds must be computed");
      return;
    }

    if (!validForPadding(op.agg(), op.combo())) {
      op.emitRemark("invalid agg/comb for use in padding");
      return;
    }

    for (unsigned i = 0; i < op.getNumTensors(); i++) {
      auto tensor = op.getTensor(i);
      auto rankedTensorType = tensor.getType().cast<mlir::RankedTensorType>();
      if (!rankedTensorType.hasStaticShape()) {
        op.emitRemark("padding cannot support dynamic memref sizes");
        return;
      }

      // Compute the largest region the source may access.
      auto bounds = computePaddingBounds(op.getSourceMap(i), *op.lowerBounds(),
                                         *op.upperBounds());

      // Check if there are any out-of-bounds reads.
      auto shape = rankedTensorType.getShape();
      assert(bounds.size() == shape.size());
      bool needsPadding = false;
      for (unsigned i = 0, e = bounds.size(); i < e; ++i) {
        if (bounds[i].min < 0 || bounds[i].max >= shape[i]) {
          needsPadding = true;
        }
      }

      // If there is no need to pad, don't bother adding an entry.
      if (!needsPadding)
        return;

      // Merge discovered padding into the recorded data.
      auto &info = toPad[tensor][op.agg()];
      info.lower.resize(bounds.size());
      info.upper.resize(bounds.size());
      for (unsigned i = 0, e = bounds.size(); i < e; ++i) {
        info.lower[i] = std::max(info.lower[i], -bounds[i].min);
        info.upper[i] = std::max(info.upper[i], bounds[i].max + 1 - shape[i]);
      }
    }
  });

  Builder builder(&getContext());
  for (const auto &kvp : toPad) {
    // Get the value which we want to pad.
    Value def = kvp.first;
    const auto &map = kvp.second;

    // If there are two conflicting ways to pad, don't pad.
    if (map.size() != 1) {
      if (Operation *op = def.getDefiningOp()) {
        op->emitRemark("padding would require multiple initialization values");
      }
      continue;
    }

    // Check if it's a block argument, and if so add an IdentOp to copy the
    // value.
    if (auto arg = def.dyn_cast<mlir::BlockArgument>()) {
      auto block = arg.getOwner();
      auto loc = block->getParentOp()->getLoc();
      OpBuilder inner(block->getParent());
      auto stub = inner.create<PlaceholderOp>(loc, arg.getType());
      // Construct an initial identity operation.
      auto ident = inner.create<eltwise::IdentOp>(loc, stub.result());
      // Replace all uses with ident (except for newly generated use).
      arg.replaceAllUsesWith(ident);
      ident.getOperation()->replaceUsesOfWith(stub, arg);
      stub.erase();
      // Now use ident for all further work.
      def = ident;
    }

    // Get defining operation (should now always work).
    Operation *op = def.getDefiningOp();
    assert(op);

    // Attach attributes specifying the padding details.
    AggregationKind agg = map.begin()->first;
    const auto &info = map.begin()->second;
    op->setAttr("padType",
                builder.getI64IntegerAttr(static_cast<int64_t>(agg)));
    op->setAttr("padLower", builder.getI64ArrayAttr(info.lower));
    op->setAttr("padUpper", builder.getI64ArrayAttr(info.upper));
  }
}

llvm::Optional<PaddingInfo> getPaddingInfo(Operation *op) {
  auto padType = op->getAttrOfType<IntegerAttr>("padType");
  if (!padType)
    return llvm::None;
  auto agg = util::symbolizeAggregationKind(padType.getInt());
  if (!agg)
    return llvm::None;

  PaddingInfo ret{*agg};
  auto padLower = op->getAttrOfType<ArrayAttr>("padLower");
  if (!padLower)
    return llvm::None;
  for (auto attr : padLower.getAsRange<IntegerAttr>()) {
    ret.lower.push_back(attr.getInt());
  }

  auto padUpper = op->getAttrOfType<ArrayAttr>("padUpper");
  if (!padUpper)
    return llvm::None;
  for (auto attr : padUpper.getAsRange<IntegerAttr>()) {
    ret.upper.push_back(attr.getInt());
  }

  return ret;
}

std::unique_ptr<mlir::Pass> createPadPass() {
  return std::make_unique<PadPass>();
}

} // namespace pmlc::dialect::tile
