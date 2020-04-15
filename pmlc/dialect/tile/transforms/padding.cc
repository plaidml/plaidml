// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/transforms/padding.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"

namespace pmlc::dialect::tile {

using mlir::AffineExpr;
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

// Helper function
static IntegerSet makeConstraintSet(size_t numDims, ArrayRef<AffineExpr> cons) {
  auto allFalse = llvm::SmallVector<bool, 10>(cons.size());
  return IntegerSet::get(numDims, 0, cons, allFalse);
}

struct PadPass : public PadBase<PadPass> {
  void runOnFunction() final;
};

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

  // Finally, remove unnecessary constraints
  func.walk([&](ContractionOp op) {
    // Early exit if no constraints
    if (!op.cons())
      return;
    auto numDims = op.sink().getNumDims();
    // Otherwise, we will check each constraint against each input
    llvm::SmallVector<AffineExpr, 8> savedConstraints;
    llvm::SmallVector<llvm::SmallVector<AffineExpr, 10>, 2> inputs;
    // Go over all the input tensors + makey polyhedra for any padded ones
    for (size_t i = 0; i < op.getNumTensors(); i++) {
      Value in = op.getTensor(i);
      // If it's not padded, forget it.
      if (!getPaddingInfo(in.getDefiningOp()))
        continue;
      // Get the tensor and the map associated with the tensor
      auto ttype = in.getType().cast<mlir::TensorType>();
      auto map = op.getSourceMap(i);
      // The should have the same rank.
      assert(ttype.getRank() == map.getNumResults());
      // Add all of the constraints that define the valid region of a tensor
      inputs.emplace_back();
      auto &cons = inputs.back();
      for (int64_t dim = 0; dim < ttype.getRank(); dim++) {
        // Get dimension specific expression
        auto expr = map.getResult(dim);
        // Lower bound (expr >= 0)
        cons.emplace_back(expr);
        // Upper bound (expr <= (size - 1) or in canonical form:
        // (size - 1) - expr >= 0
        cons.emplace_back(-expr + (ttype.getDimSize(dim) - 1));
      }
    }
    // If no padded inputs, early exit
    if (inputs.empty())
      return;
    // Check each constraint by adding its inverse to each input polyhedra.
    // If the resulting integer set is empty, then all of the false cases are
    // outside of the valid region (i.e. in the padded region).  Thus any loads
    // when the constraint is false will load the padding value, and thus have
    // no effect on the output.  This means that removing the constraint is
    // safe, since all places it would have previously prevented will turn into
    // no-ops post padding.
    for (auto expr : op.cons()->getConstraints()) {
      // Compute the 'inverse' of the constraint.
      // The opposite of expr >= 0 is expr < 0
      // Given expr is an integer this is the same as expr + 1 <= 0 or
      // -expr - 1 >= 0
      auto inv_expr = -expr - 1;
      // Initially presume we will not keep this constraint
      bool keep = true;
      // For each input
      for (auto &cons : inputs) {
        // Temporarily add the inv_expr to the constraints
        cons.push_back(inv_expr);
        // Make into an IntegerSet, and then into FlatAffineConstraints.
        // It seems wasteful to intern a temporary integer set, but any other
        // way of doing this is also annoying given the current class structures
        auto set = makeConstraintSet(numDims, cons);
        mlir::FlatAffineConstraints fac(set);
        // Can't prove it's empty, keep constraint
        if (fac.isEmpty())
          keep = false;
        // Remove
        cons.pop_back();
        // If we've decided not to keep the constraint, don't bother considering
        // more inputs.
        if (!keep) {
          break;
        }
      }
      if (keep)
        savedConstraints.push_back(expr);
    }
    if (savedConstraints.empty()) {
      op.removeAttr(ContractionOp::getConstraintsAttrName());
    } else {
      op.setConstraints(makeConstraintSet(numDims, savedConstraints));
    }
  });
}

llvm::Optional<PaddingInfo> getPaddingInfo(Operation *op) {
  if (!op)
    return llvm::None;
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
