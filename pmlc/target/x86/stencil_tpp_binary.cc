// Copyright 2020 Intel Corporation

#include <string>

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
  std::string opName;
  template <typename OpTy>
  void maybeCaptureGeneric(Optional<pxa::StencilCapture> &capture,
                           const std::string &inName) {
    if (capture)
      return;
    using matchers::m_Any;
    Value load1, load2, reduce;

    auto pattern = m_Op<AffineYieldOp>(m_Capture(
        &reduce, pxa::m_PxaReduceOp(
                     AtomicRMWKind::assign,
                     m_Op<OpTy>(m_Capture(&load1, m_Op<pxa::PxaLoadOp>()),
                                m_Capture(&load2, m_Op<pxa::PxaLoadOp>())),
                     m_Any())));

    Operation *yield = op.getBody()->getTerminator();
    if (!matchPattern(yield, pattern))
      return;

    capture = pxa::StencilCapture{{reduce}, {load1, load2}};
    this->opName = inName;
  }

  Optional<pxa::StencilCapture> capture() {
    Optional<pxa::StencilCapture> ret;

    maybeCaptureGeneric<AddFOp>(ret, "tpp_add");
    maybeCaptureGeneric<MulFOp>(ret, "tpp_mul");
    maybeCaptureGeneric<AddFOp>(ret, "tpp_sub");
    maybeCaptureGeneric<DivFOp>(ret, "tpp_div");

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
    auto inputOp1 =
        cast<pxa::PxaLoadOp>(stencil.values[1].value.getDefiningOp());

    auto inputOp2 =
        cast<pxa::PxaLoadOp>(stencil.values[2].value.getDefiningOp());

    SmallVector<Value> inputIndices, outputIndices;
    Optional<TppOperand> output =
        getTppOperand(outputOp, op->getBlock(), stencil.indexes, outputIndices);
    if (!output)
      return;

    IVLOG(3, "stencil tpp binary: output parsed  ");

    Optional<TppOperand> input1 =
        getTppOperand(inputOp1, op->getBlock(), stencil.indexes, inputIndices);
    if (!input1)
      return;

    Optional<TppOperand> input2 =
        getTppOperand(inputOp2, op->getBlock(), stencil.indexes, inputIndices);
    if (!input2)
      return;

    ArrayAttr outputAccessMaps =
        builder.getAffineMapArrayAttr({output->accessMap});
    ArrayAttr outputTileMaps = builder.getAffineMapArrayAttr({output->tileMap});

    ArrayAttr inputAccessMaps =
        builder.getAffineMapArrayAttr({input1->accessMap, input2->accessMap});
    ArrayAttr inputTileMaps =
        builder.getAffineMapArrayAttr({input1->tileMap, input2->tileMap});

    ArrayAttr reductions =
        builder.getI64ArrayAttr({static_cast<int64_t>(outputOp.agg())});

    auto genericOp = builder.create<pxa::PxaGenericOp>(
        op.getLoc(), output->memref.getType(),
        /*inputs=*/ArrayRef<Value>{input1->memref, input2->memref},
        /*outputs=*/ArrayRef<Value>{output->memref},
        /*inputIndices=*/inputIndices,
        /*outputIndices=*/outputIndices,
        /*inputAccessMaps=*/inputAccessMaps,
        /*inputTileMaps=*/inputTileMaps,
        /*outputAccessMaps=*/outputAccessMaps,
        /*outputTileMaps=*/outputTileMaps,
        /*kernel=*/builder.getStringAttr(this->opName),
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
                        [](int64_t stride) { return stride > 1; }, // input
                        [](int64_t stride) { return stride > 1; }, // input
                    }},
                pxa::StencilIndexRequirement{
                    /*idxName=*/"eltwise_j",
                    /*tilingGenerator=*/pxa::ExactRangeGenerator(),
                    pxa::IndexStridePredicates{
                        [](int64_t stride) { return stride == 1; }, // output
                        [](int64_t stride) { return stride == 1; }, // input
                        [](int64_t stride) { return stride == 1; }, // input
                    }},
            }) {}
};

} // namespace

struct StencilTppBinaryPass
    : public StencilTppBinaryBase<StencilTppBinaryPass> {
  void runOnFunction() final {
    getFunction().walk([](AffineParallelOp op) {
      // if (op.getIVs().size() == 2) {
      StencilImpl stencil(op);
      stencil.performStenciling();
      //}
    });
  }
};

std::unique_ptr<Pass> createStencilTppBinaryPass() {
  return std::make_unique<StencilTppBinaryPass>();
}

} // namespace pmlc::target::x86
