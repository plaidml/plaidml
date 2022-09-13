// Copyright 2020 Intel Corporation
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "pmlc/dialect/pxa/ir/matchers.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/autotile.h"
#include "pmlc/dialect/pxa/transforms/stencil.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/target/x86/passes.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/matchers.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT

namespace pmlc::target::x86 {

namespace pxa = dialect::pxa;
namespace stdx = dialect::stdx;

namespace {

struct GemmOperand {
  Value memref;
  AffineMap accessMap;
  AffineMap tileMap;

  template <typename TOp>
  GemmOperand(TOp op, ArrayRef<BlockArgument> idxs,
              SmallVectorImpl<Value> &mapOperands)
      : memref(op.getMemRef()), accessMap(op.getAffineMap()),
        tileMap(pxa::makeTileMap(op.getContext(), op.getAffineMap(),
                                 op.getMapOperands(), idxs)) {
    mapOperands.append(op.getMapOperands().begin(), op.getMapOperands().end());
  }
};

double getStride(ArrayRef<int64_t> tileSizes, int tiledIdxCount,
                 DenseMap<mlir::BlockArgument, int64_t> strides,
                 SmallVector<mlir::BlockArgument, 8> indexes) {
  for (size_t j = 0; j < tiledIdxCount; j++) {
    for (const auto &kvp : strides) {
      if (indexes[j] == kvp.first) {
        return static_cast<double>(kvp.second);
      }
    }
  }
  return 0.0;
}

class StencilImpl : public pxa::StencilBase {
private:
  StringRef opName;

  template <typename OpTy>
  void maybeCaptureGeneric(Optional<pxa::StencilCapture> &capture,
                           StringRef inName) {
    if (capture)
      return;
    AffineValueMap ranges = util::getRangesValueMap(op);
    if (op.getIVs().size() == 2) {
      auto constExpr = ranges.getResult(0).dyn_cast<AffineConstantExpr>();
      if (constExpr && constExpr.getValue() == 1)
        return;
    }

    using matchers::m_Any;

    Value load, reduce;
    auto pattern = m_Op<AffineYieldOp>(m_Capture(
        &reduce,
        pxa::m_PxaReduceOp(arith::AtomicRMWKind::assign,
                           m_Op<OpTy>(m_Capture(&load, m_Op<pxa::PxaLoadOp>())),
                           m_Any())));

    Operation *yield = op.getBody()->getTerminator();
    if (!matchPattern(yield, pattern))
      return;

    if (!load.getType().isF32() ||
        !reduce.getType().cast<MemRefType>().getElementType().isF32())
      return;
    auto source = cast<pxa::PxaLoadOp>(load.getDefiningOp()).memref();
    if (!source.isa<BlockArgument>()) {
      // If the definition of load's source is in "op", it is too complex to
      // stencil
      auto defOp = source.getDefiningOp();
      while (!isa<func::FuncOp>(defOp)) {
        if (defOp == op.getOperation())
          return;
        defOp = defOp->getParentOp();
      }
    }

    capture = pxa::StencilCapture{{reduce}, {load}};
    this->opName = inName;
  }

  void maybeCaptureIdentityOp(Optional<pxa::StencilCapture> &capture,
                              StringRef inName) {
    if (capture)
      return;
    AffineValueMap ranges = util::getRangesValueMap(op);
    if (op.getIVs().size() == 2) {
      auto constExpr = ranges.getResult(0).dyn_cast<AffineConstantExpr>();
      if (constExpr && constExpr.getValue() == 1)
        return;
    }

    using matchers::m_Any;

    Value load, reduce;
    auto pattern = m_Op<AffineYieldOp>(m_Capture(
        &reduce,
        pxa::m_PxaReduceOp(arith::AtomicRMWKind::assign,
                           m_Capture(&load, m_Op<pxa::PxaLoadOp>()), m_Any())));

    Operation *yield = op.getBody()->getTerminator();
    if (!matchPattern(yield, pattern))
      return;
    if (!load.getType().isF32() ||
        !reduce.getType().cast<MemRefType>().getElementType().isF32())
      return;
    auto source = cast<pxa::PxaLoadOp>(load.getDefiningOp()).memref();
    if (!source.isa<BlockArgument>()) {
      // If the definition of load's source is in "op", it is too complex to
      // stencil
      auto defOp = source.getDefiningOp();
      while (!isa<func::FuncOp>(defOp)) {
        if (defOp == op.getOperation())
          return;
        defOp = defOp->getParentOp();
      }
    }

    capture = pxa::StencilCapture{{reduce}, {load}};
    this->opName = inName;
  }

  void maybeCaptureReduceOp(Optional<pxa::StencilCapture> &capture,
                            StringRef inName, arith::AtomicRMWKind agg) {
    if (capture)
      return;
    if (op.getIVs().size() > 2)
      return;
    AffineValueMap ranges = util::getRangesValueMap(op);
    auto constExpr = ranges.getResult(0).dyn_cast<AffineConstantExpr>();
    if (constExpr && constExpr.getValue() > 1)
      return;

    using matchers::m_Any;

    Value load, reduce;
    auto pattern = m_Op<AffineYieldOp>(m_Capture(
        &reduce, pxa::m_PxaReduceOp(
                     agg, m_Capture(&load, m_Op<pxa::PxaLoadOp>()), m_Any())));

    Operation *yield = op.getBody()->getTerminator();
    if (!matchPattern(yield, pattern))
      return;
    if (!load.getType().isF32() ||
        !reduce.getType().cast<MemRefType>().getElementType().isF32())
      return;
    auto source = cast<pxa::PxaLoadOp>(load.getDefiningOp()).memref();
    if (!source.isa<BlockArgument>()) {
      // If the definition of load's source is in "op", it is too complex to
      // stencil
      auto defOp = source.getDefiningOp();
      while (!isa<func::FuncOp>(defOp)) {
        if (defOp == op.getOperation())
          return;
        defOp = defOp->getParentOp();
      }
    }

    // TODO: remove the constraint of memRefType
    auto type = cast<pxa::PxaReduceOp>(reduce.getDefiningOp()).getMemRefType();
    if (type.getShape().size() != 2 && type.getShape().size() != 4)
      return;

    capture = pxa::StencilCapture{{reduce}, {load}};
    this->opName = inName;
  }

  Optional<pxa::StencilCapture> capture() {
    Optional<pxa::StencilCapture> ret;
    maybeCaptureGeneric<stdx::ReluOp>(ret, "tpp_relu");
    maybeCaptureGeneric<math::TanhOp>(ret, "tpp_tanh");
    maybeCaptureGeneric<math::ExpOp>(ret, "tpp_exp");
    maybeCaptureIdentityOp(ret, "tpp_identity");
    maybeCaptureReduceOp(ret, "tpp_add_reduce", arith::AtomicRMWKind::addf);
    maybeCaptureReduceOp(ret, "tpp_max_reduce", arith::AtomicRMWKind::maxf);
    // maybeCaptureReduceOp(ret, "tpp_mul_reduce", arith::AtomicRMWKind::mulf);
    return ret;
  }

  std::pair<double, double> getCost(const pxa::StencilOption &stencil,
                                    ArrayRef<int64_t> tileSizes) {
    int64_t tiledIdxCount = getTiledIdxCount();
    double inputStride =
        getStride(tileSizes, tiledIdxCount,
                  stencil.values[1].strideInfo.strides, stencil.indexes);
    double outputStride =
        getStride(tileSizes, tiledIdxCount,
                  stencil.values[0].strideInfo.strides, stencil.indexes);

    return std::make_pair((inputStride + outputStride), 0.0);
  }

  void transform(const pxa::StencilOption &stencil,
                 ArrayRef<int64_t> tileSizes) {
    auto outputOp =
        cast<pxa::PxaReduceOp>(stencil.values[0].value.getDefiningOp());
    auto inputOp =
        cast<pxa::PxaLoadOp>(stencil.values[1].value.getDefiningOp());

    OpBuilder builder = op.getBodyBuilder();
    SmallVector<Value> inputIndices, outputIndices;

    GemmOperand output(outputOp, stencil.indexes, outputIndices);

    GemmOperand input(inputOp, stencil.indexes, inputIndices);

    SmallVector<int64_t, 8> steps = op.getSteps();
    for (size_t i = 0; i < steps.size(); i++) {
      BlockArgument idx = op.getBody()->getArgument(i);
      int64_t idxRange = getIdxRange(idx);

      for (size_t j = 0; j < getTiledIdxCount(); j++) {
        if (stencil.indexes[j] == idx) {
          steps[i] *= tileSizes[j];
        }
      }
    }
    op.setSteps(steps);

    ArrayAttr outputAccessMaps =
        builder.getAffineMapArrayAttr({output.accessMap});
    ArrayAttr outputTileMaps = builder.getAffineMapArrayAttr({output.tileMap});

    ArrayAttr inputAccessMaps =
        builder.getAffineMapArrayAttr({input.accessMap});
    ArrayAttr inputTileMaps = builder.getAffineMapArrayAttr({input.tileMap});

    ArrayAttr reductions =
        builder.getI64ArrayAttr({static_cast<int64_t>(outputOp.agg())});

    auto genericOp = builder.create<pxa::PxaGenericOp>(
        op.getLoc(), output.memref.getType(),
        /*inputs=*/ArrayRef<Value>{input.memref},
        /*outputs=*/ArrayRef<Value>{output.memref},
        /*inputIndices=*/inputIndices,
        /*outputIndices=*/outputIndices,
        /*inputAccessMaps=*/inputAccessMaps,
        /*inputTileMaps=*/inputTileMaps,
        /*outputAccessMaps=*/outputAccessMaps,
        /*outputTileMaps=*/outputTileMaps,
        /*kernel=*/builder.getStringAttr(opName),
        /*tile=*/builder.getI64ArrayAttr(tileSizes),
        /*reductions=*/reductions);

    outputOp.result().replaceAllUsesWith(genericOp.getResult(0));
    outputOp.erase();
  }

public:
  explicit StencilImpl(AffineParallelOp op)
      : StencilBase(
            op,
            {
                pxa::StencilIndexRequirement{
                    /*idxName=*/"eltwise_i",
                    /*tilingGenerator=*/pxa::ExactRangeGenerator(),
                    pxa::IndexStridePredicates{
                        [](int64_t stride) {
                          return (stride == 0 || stride > 1);
                        }, // output
                        [](int64_t stride) {
                          return (stride == 0 || stride > 1);
                        }, // input
                    }},
                pxa::StencilIndexRequirement{
                    /*idxName=*/"eltwise_j",
                    /*tilingGenerator=*/pxa::ExactRangeGenerator(),
                    pxa::IndexStridePredicates{
                        [](int64_t stride) {
                          return (stride == 1 || stride == 0);
                        },                                            // output
                        [](int64_t stride) { return (stride >= 0); }, // input
                    }},
            }) {}
};

} // namespace

void splitReducePattern(AffineParallelOp op) {
  if (op.getIVs().size() < 2)
    return;
  using matchers::m_Any;

  Value load, reduce;
  Operation *yield = op.getBody()->getTerminator();
  auto add_pattern = m_Op<AffineYieldOp>(m_Capture(
      &reduce,
      pxa::m_PxaReduceOp(arith::AtomicRMWKind::addf,
                         m_Capture(&load, m_Op<pxa::PxaLoadOp>()), m_Any())));
  auto max_pattern = m_Op<AffineYieldOp>(m_Capture(
      &reduce,
      pxa::m_PxaReduceOp(arith::AtomicRMWKind::maxf,
                         m_Capture(&load, m_Op<pxa::PxaLoadOp>()), m_Any())));
  // auto mul_pattern = m_Op<AffineYieldOp>(m_Capture(
  //     &reduce,
  //     pxa::m_PxaReduceOp(arith::AtomicRMWKind::mulf,
  //                        m_Capture(&load, m_Op<pxa::PxaLoadOp>()),
  //                        m_Any())));
  if (!matchPattern(yield, add_pattern) && !matchPattern(yield, max_pattern))
    return;

  // Make builder
  OpBuilder builder(op.getBody(), op.getBody()->begin());
  Block *outerBody = op.getBody();
  size_t rank = op.lowerBoundsMap().getNumResults();
  // Make the maps for the inner parallel
  SmallVector<AffineMap, 8> lbMaps;
  SmallVector<AffineMap, 8> ubMaps;
  SmallVector<int64_t, 8> newSteps;
  auto zeroExpr = builder.getAffineConstantExpr(0);
  auto oneExpr = builder.getAffineConstantExpr(1);
  auto steps = op.getSteps();
  for (size_t i = 0; i < rank - 1; i++) {
    newSteps.push_back(steps[i]);
    lbMaps.push_back(AffineMap::get(rank, 0, zeroExpr));
    ubMaps.push_back(AffineMap::get(rank, 0, oneExpr));
  }
  AffineValueMap ranges = util::getRangesValueMap(op);
  auto constExpr = ranges.getResult(rank - 1).dyn_cast<AffineConstantExpr>();
  newSteps.push_back(constExpr.getValue());
  lbMaps.push_back(
      AffineMap::get(rank, 0, op.lowerBoundsMap().getResult(rank - 1)));
  ubMaps.push_back(
      AffineMap::get(rank, 0, op.upperBoundsMap().getResult(rank - 1)));
  auto outerIdxs = outerBody->getArguments();
  // Make the inner parallel for (above all other code);
  SmallVector<arith::AtomicRMWKind, 8> reductions;
  for (Attribute attr : op.reductions()) {
    auto intAttr = attr.dyn_cast<IntegerAttr>();
    reductions.push_back(*arith::symbolizeAtomicRMWKind(intAttr.getInt()));
  }
  auto inner = builder.create<AffineParallelOp>(
      op.getLoc(), op.getResultTypes(), reductions, lbMaps, outerIdxs, ubMaps,
      outerIdxs, steps);
  // Splice instructions into the interior
  auto &innerLoopOps = inner.getBody()->getOperations();
  auto &outerLoopOps = outerBody->getOperations();
  innerLoopOps.splice(std::prev(innerLoopOps.end()), outerLoopOps,
                      std::next(outerLoopOps.begin(), 1), outerLoopOps.end());
  // Replace old indices with new indices
  auto innerIdxs = inner.getBody()->getArguments();
  outerIdxs[rank - 1].replaceAllUsesWith(innerIdxs[rank - 1]);
  unsigned numIdxs = inner.lowerBoundsMap().getNumInputs();
  for (unsigned i = 0; i < numIdxs; ++i) {
    inner.setOperand(i, outerIdxs[i]);
    inner.setOperand(i + numIdxs, outerIdxs[i]);
  }
  // Add a return of the values of the inner to the outer
  builder.setInsertionPointToEnd(op.getBody());
  builder.create<AffineYieldOp>(op.getLoc(), inner.getResults());
  // Update outer step size
  op.setSteps(newSteps);
}

// For 1D stenciling, we still map to 2D TPPs. We can make 2D TPPs do 1D
// by having the outer dim being 1.
void addSingleIteration(AffineParallelOp op) {
  if (op.getIVs().size() != 1)
    return;

  Block *body = op.getBody();
  auto builder = OpBuilder::atBlockBegin(body);
  auto zeroExpr = builder.getAffineConstantExpr(0);
  auto oneExpr = builder.getAffineConstantExpr(1);
  SmallVector<AffineExpr, 6> newLowerBounds;
  SmallVector<AffineExpr, 6> newUpperBounds;
  SmallVector<int64_t, 6> newSteps;
  SmallVector<int32_t> groups;
  auto steps = op.getSteps();
  // Add a single iteration
  newLowerBounds.push_back(zeroExpr);
  newUpperBounds.push_back(oneExpr);
  newSteps.push_back(1);
  groups.push_back(1);
  // Keep the original args
  body->addArguments(body->getArguments()[0].getType(), op.getLoc());
  for (unsigned i = 0, e = steps.size(); i < e; ++i) {
    int64_t step = steps[i];
    newLowerBounds.push_back(op.lowerBoundsMap().getResult(i));
    newUpperBounds.push_back(op.upperBoundsMap().getResult(i));
    newSteps.push_back(step);
    groups.push_back(1);
  }
  auto newArgs = body->getArguments();
  newArgs[0].replaceAllUsesWith(newArgs[1]);

  // Update maps
  auto newLower = AffineMap::get(op.lowerBoundsMap().getNumDims(),
                                 op.lowerBoundsMap().getNumSymbols(),
                                 newLowerBounds, op.getContext());
  auto newUpper = AffineMap::get(op.upperBoundsMap().getNumDims(),
                                 op.upperBoundsMap().getNumSymbols(),
                                 newUpperBounds, op.getContext());
  op.lowerBoundsMapAttr(AffineMapAttr::get(newLower));
  op.lowerBoundsGroupsAttr(builder.getI32TensorAttr(groups));
  op.upperBoundsMapAttr(AffineMapAttr::get(newUpper));
  op.upperBoundsGroupsAttr(builder.getI32TensorAttr(groups));
  op.setSteps(newSteps);
}

struct StencilTppUnaryPass : public StencilTppUnaryBase<StencilTppUnaryPass> {
  void runOnOperation() final {
    getOperation().walk(splitReducePattern);
    getOperation().walk(addSingleIteration);
    getOperation().walk([](AffineParallelOp op) {
      if (op.getIVs().size() >= 2) {
        StencilImpl stencil(op);
        stencil.performStenciling();
      }
    });
  }
};

std::unique_ptr<Pass> createStencilTppUnaryPass() {
  return std::make_unique<StencilTppUnaryPass>();
}

} // namespace pmlc::target::x86
