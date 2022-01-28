// Copyright 2020 Intel Corporation
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

    using matchers::m_Any;

    Value load, reduce;
    auto pattern = m_Op<AffineYieldOp>(m_Capture(
        &reduce,
        pxa::m_PxaReduceOp(AtomicRMWKind::assign,
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
      while (!isa<FuncOp>(defOp)) {
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

    using matchers::m_Any;

    Value load, reduce;
    auto pattern = m_Op<AffineYieldOp>(m_Capture(
        &reduce,
        pxa::m_PxaReduceOp(AtomicRMWKind::assign,
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
      while (!isa<FuncOp>(defOp)) {
        if (defOp == op.getOperation())
          return;
        defOp = defOp->getParentOp();
      }
    }

    capture = pxa::StencilCapture{{reduce}, {load}};
    this->opName = inName;
  }

  Optional<pxa::StencilCapture> capture() {
    Optional<pxa::StencilCapture> ret;
    maybeCaptureGeneric<stdx::ReluOp>(ret, "tpp_relu");
    maybeCaptureGeneric<math::TanhOp>(ret, "tpp_tanh");
    maybeCaptureGeneric<math::ExpOp>(ret, "tpp_exp");
    maybeCaptureIdentityOp(ret, "tpp_identity");
    return ret;
  }

  double getCost(const pxa::StencilOption &stencil,
                 ArrayRef<int64_t> tileSizes) {
    int64_t tiledIdxCount = getTiledIdxCount();
    double inputStride =
        getStride(tileSizes, tiledIdxCount,
                  stencil.values[1].strideInfo.strides, stencil.indexes);
    double outputStride =
        getStride(tileSizes, tiledIdxCount,
                  stencil.values[0].strideInfo.strides, stencil.indexes);

    return (inputStride + outputStride);
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
                        [](int64_t stride) { return stride > 1; }, // output
                        [](int64_t stride) {
                          return (stride == 0 || stride > 1);
                        }, // input
                    }},
                pxa::StencilIndexRequirement{
                    /*idxName=*/"eltwise_j",
                    /*tilingGenerator=*/pxa::ExactRangeGenerator(),
                    pxa::IndexStridePredicates{
                        [](int64_t stride) { return stride == 1; }, // output
                        [](int64_t stride) {
                          return (stride == 1 || stride == 0);
                        }, // input
                    }},
            }) {}
};

} // namespace

struct StencilTppUnaryPass : public StencilTppUnaryBase<StencilTppUnaryPass> {
  void runOnFunction() final {
    getFunction().walk([](AffineParallelOp op) {
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
