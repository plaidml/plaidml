// Copyright 2019, Intel Corporation

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
  size_t idxs = access.getNumDims();
  size_t size = access.getNumResults();
  assert(lower.getNumResults() == idxs);
  assert(upper.getNumResults() == idxs);
  std::vector<BoundRange> dimRanges;
  for (size_t i = 0; i < idxs; i++) {
    int64_t lowerBoundInt =
        lower.getResult(i).cast<mlir::AffineConstantExpr>().getValue();
    int64_t upperBoundInt =
        upper.getResult(i).cast<mlir::AffineConstantExpr>().getValue();
    dimRanges.emplace_back(lowerBoundInt, upperBoundInt);
  }
  ComputeRangeVisitor visitor(std::move(dimRanges));
  llvm::SmallVector<BoundRange, 4> outRanges;
  for (size_t i = 0; i < size; i++) {
    outRanges.push_back(visitor.visit(access.getResult(i)));
  }
  return outRanges;
}

struct PaddingInfo {
  llvm::SmallVector<int64_t, 4> padBelow;
  llvm::SmallVector<int64_t, 4> padAbove;
};

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
    if (!op.lowerBounds().hasValue() || !op.upperBounds().hasValue()) {
      op.emitRemark("contraction bounds must be computed");
      return;
    }
    if (!validForPadding(op.agg(), op.combo())) {
      op.emitRemark("invalid agg/comb for use in padding");
      return;
    }
    for (size_t i = 0; i < op.getNumTensors(); i++) {
      // Compute the largest region the source may access.
      auto bounds = computePaddingBounds(op.getSourceMap(i), *op.lowerBounds(),
                                         *op.upperBounds());
      // Check if there are any out-of-bounds reads.
      auto shape =
          op.getTensor(i).getType().cast<mlir::RankedTensorType>().getShape();
      assert(bounds.size() == shape.size());
      bool needsPadding = false;
      for (size_t i = 0; i < bounds.size(); i++) {
        if (shape[i] == mlir::RankedTensorType::kDynamicSize) {
          op.emitRemark("padding cannot support dynamic memref sizes");
          return;
        }
        if (bounds[i].min < 0 || bounds[i].max >= shape[i]) {
          needsPadding = true;
        }
      }
      // If there is no need to pad, don't bother adding an entry.
      if (!needsPadding)
        return;
      // Merge discovered padding into the recorded data.
      auto &info = toPad[op.getTensor(i)][op.agg()];
      info.padBelow.resize(bounds.size());
      info.padAbove.resize(bounds.size());
      for (size_t i = 0; i < bounds.size(); i++) {
        info.padBelow[i] = std::max(info.padBelow[i], -bounds[i].min);
        info.padAbove[i] =
            std::max(info.padAbove[i], bounds[i].max + 1 - shape[i]);
      }
    }
  });
  Builder builder(&getContext());
  for (const auto &kvp : toPad) {
    // Get the value which we want to pad.
    Value def = kvp.first;
    // If there are two conflicting ways to pad, don't pad.
    if (kvp.second.size() != 1) {
      if (Operation *op = def.getDefiningOp()) {
        op->emitRemark("padding would require multiple initialization values");
      }
      continue;
    }
    // Check if it's a block argument, and if so add an IdentOp to copy the
    // value.
    if (auto arg = def.dyn_cast<mlir::BlockArgument>()) {
      // Construct an initial identity operation.
      auto block = arg.getOwner();
      OpBuilder inner(block, block->begin());
      auto ident =
          inner.create<eltwise::IdentOp>(block->getParentOp()->getLoc(), def);
      // Replace all uses with ident (except for newly generated use).
      // TODO: This seems like the wrong way to do things?
      auto use = def.use_begin();
      while (use != def.use_end()) {
        // Copy and increment since we will destroy use.
        auto newUse = use;
        newUse++;
        // Change if not the IdentOp itself.
        if (use->getOwner() != ident.getOperation()) {
          use->set(ident);
        }
        use = newUse;
      }
      // Now use ident for all further work.
      def = ident;
    }
    // Get defining operation (should now always work).
    Operation *op = def.getDefiningOp();
    assert(op);

    // Attach attributes specifying the padding details.
    AggregationKind agg = kvp.second.begin()->first;
    op->setAttr("padType",
                builder.getI64IntegerAttr(static_cast<int64_t>(agg)));
    const auto &info = kvp.second.begin()->second;
    op->setAttr("padBelow", builder.getI64ArrayAttr(info.padBelow));
    op->setAttr("padAbove", builder.getI64ArrayAttr(info.padAbove));
  }
}

std::unique_ptr<mlir::Pass> createPadPass() {
  return std::make_unique<PadPass>();
}

} // namespace pmlc::dialect::tile
