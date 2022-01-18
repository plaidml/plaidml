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

struct TppOperand {
  Value memref;
  AffineMap accessMap;
  AffineMap tileMap;
};

template <typename TOp>
Optional<TppOperand> getTppOperand(TOp op, Block *block,
                                   ArrayRef<BlockArgument> idxs,
                                   SmallVectorImpl<Value> &mapOperands) {
  Optional<pxa::RelativeAccessPattern> rap =
      pxa::computeRelativeAccess(op, block);
  if (!rap)
    return None;

  AffineValueMap outerValueMap =
      pxa::convertToValueMap(op.getContext(), rap->outer);
  mapOperands.append(outerValueMap.getOperands().begin(),
                     outerValueMap.getOperands().end());

  AffineMap tileMap = pxa::makeTileMap(op.getContext(), op.getAffineMap(),
                                       op.getMapOperands(), idxs);

  return TppOperand{op.getMemRef(), outerValueMap.getAffineMap(), tileMap};
}

class StencilImpl : public pxa::StencilBase {
private:
  StringRef opName;
  Value scalar;
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
                           m_Op<OpTy>(m_Capture(&load, m_Op<ConstantOp>())),
                           m_Any())));

    Operation *yield = op.getBody()->getTerminator();
    if (!matchPattern(yield, pattern))
      return;

    if (!load.getType().isF32() ||
        !reduce.getType().cast<MemRefType>().getElementType().isF32())
      return;

    capture = pxa::StencilCapture{{reduce}, {}};
    this->opName = inName;
    this->scalar = load;
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
                           m_Capture(&load, m_Op<ConstantOp>()), m_Any())));

    Operation *yield = op.getBody()->getTerminator();
    if (!matchPattern(yield, pattern))
      return;

    if (!load.getType().isF32() ||
        !reduce.getType().cast<MemRefType>().getElementType().isF32())
      return;

    capture = pxa::StencilCapture{{reduce}, {}};
    this->opName = inName;
    this->scalar = load;
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
    return 0.0;
  }

  void transform(const pxa::StencilOption &stencil,
                 ArrayRef<int64_t> tileSizes) {
    OpBuilder builder(op);

    auto outputOp =
        cast<pxa::PxaReduceOp>(stencil.values[0].value.getDefiningOp());

    SmallVector<Value> inputIndices, outputIndices;
    Optional<TppOperand> output =
        getTppOperand(outputOp, op->getBlock(), stencil.indexes, outputIndices);
    if (!output)
      return;

    ArrayAttr outputAccessMaps =
        builder.getAffineMapArrayAttr({output->accessMap});
    ArrayAttr outputTileMaps = builder.getAffineMapArrayAttr({output->tileMap});

    ArrayAttr inputAccessMaps = builder.getAffineMapArrayAttr({});
    ArrayAttr inputTileMaps = builder.getAffineMapArrayAttr({});

    ArrayAttr reductions =
        builder.getI64ArrayAttr({static_cast<int64_t>(outputOp.agg())});

    auto genericOp = builder.create<pxa::PxaGenericOp>(
        op.getLoc(), output->memref.getType(),
        /*inputs=*/ArrayRef<Value>{scalar},
        /*outputs=*/ArrayRef<Value>{output->memref},
        /*inputIndices=*/inputIndices,
        /*outputIndices=*/outputIndices,
        /*inputAccessMaps=*/inputAccessMaps,
        /*inputTileMaps=*/inputTileMaps,
        /*outputAccessMaps=*/outputAccessMaps,
        /*outputTileMaps=*/outputTileMaps,
        /*kernel=*/builder.getStringAttr(opName),
        /*tile=*/builder.getI64ArrayAttr(tileSizes),
        /*reductions=*/reductions);

    op.getResult(0).replaceAllUsesWith(genericOp.getResult(0));
    op.erase();
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
                    }},
                pxa::StencilIndexRequirement{
                    /*idxName=*/"eltwise_j",
                    /*tilingGenerator=*/pxa::ExactRangeGenerator(),
                    pxa::IndexStridePredicates{
                        [](int64_t stride) { return stride == 1; }, // output
                    }},
            }) {}
};

} // namespace

struct StencilTppUnaryScalarPass
    : public StencilTppUnaryScalarBase<StencilTppUnaryScalarPass> {
  void runOnFunction() final {
    getFunction().walk([](AffineParallelOp op) {
      if (op.getIVs().size() == 2) {
        StencilImpl stencil(op);
        stencil.performStenciling();
      }
    });
  }
};

std::unique_ptr<Pass> createStencilTppUnaryScalarPass() {
  return std::make_unique<StencilTppUnaryScalarPass>();
}

} // namespace pmlc::target::x86
