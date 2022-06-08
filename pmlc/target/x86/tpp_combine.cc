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
#include "pmlc/util/memuse.h"
#include "pmlc/util/util.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir; // NOLINT

namespace pmlc::target::x86 {

namespace pxa = dialect::pxa;
namespace stdx = dialect::stdx;

namespace {

class TppCombineImpl {
public:
  SmallVector<pxa::PxaGenericOp, 3> maybeCaptureGemm(AffineParallelOp op) {
    using matchers::m_Any;

    Value *reluOp = new Value();
    Value *gemmOp = new Value();
    Value *identityOp = new Value();

    auto opPattern = m_Op<AffineYieldOp>(m_Capture(
        reluOp, m_Op<pxa::PxaGenericOp>())); // m_Capture(gemmOp,
                                             // m_Op<pxa::PxaGenericOp>()))));
    auto affineYield = op.getBody()->getTerminator();
    if (!matchPattern(affineYield, opPattern)) {
      return {NULL, NULL, NULL};
    }
    pxa::PxaGenericOp pxaReluOp =
        dyn_cast<pxa::PxaGenericOp>(reluOp->getDefiningOp());
    if (pxaReluOp.kernel().str() != "tpp_relu") {
      return {NULL, NULL, NULL};
    }
    auto gemmOpPattern = m_Capture(gemmOp, m_Op<pxa::PxaGenericOp>());
    if (!matchOperandOrValueAtIndex(pxaReluOp, 0, gemmOpPattern)) {
      return {NULL, NULL, NULL};
    }
    pxa::PxaGenericOp pxaGemmOp =
        dyn_cast<pxa::PxaGenericOp>(gemmOp->getDefiningOp());
    if (pxaGemmOp.kernel().str() != "tpp_gemm") {
      return {NULL, NULL, NULL};
    }
    auto identityOpPattern = m_Capture(identityOp, m_Op<pxa::PxaGenericOp>());
    if (!matchOperandOrValueAtIndex(pxaGemmOp, 2, identityOpPattern)) {
      return {NULL, NULL, NULL};
    }
    pxa::PxaGenericOp pxaIdentityOp =
        dyn_cast<pxa::PxaGenericOp>(identityOp->getDefiningOp());
    if (pxaIdentityOp.kernel().str() != "tpp_identity") {
      return {NULL, NULL, NULL};
    }
    return {pxaReluOp, pxaGemmOp, pxaIdentityOp};
  }

  SmallVector<pxa::PxaGenericOp, 6>
  maybeCaptureChainedGemms(AffineParallelOp op) {
    using matchers::m_Any;
    Value *reluOp = new Value();
    Value *gemmOp0 = new Value();
    Value *identityOp0 = new Value();
    Value *gemmOp1 = new Value();
    Value *identityOp1 = new Value();
    Value *addOp = new Value();

    auto opPattern = m_Op<AffineYieldOp>(m_Capture(
        reluOp, m_Op<pxa::PxaGenericOp>())); // m_Capture(gemmOp,
                                             // m_Op<pxa::PxaGenericOp>()))));
    auto affineYield = op.getBody()->getTerminator();
    if (!matchPattern(affineYield, opPattern)) {
      return {NULL, NULL, NULL, NULL, NULL, NULL};
    }
    pxa::PxaGenericOp pxaReluOp =
        dyn_cast<pxa::PxaGenericOp>(reluOp->getDefiningOp());
    if (pxaReluOp.kernel().str() != "tpp_relu") {
      return {NULL, NULL, NULL, NULL, NULL, NULL};
    }
    auto addOpPattern = m_Capture(addOp, m_Op<pxa::PxaGenericOp>());
    if (!matchOperandOrValueAtIndex(pxaReluOp, 0, addOpPattern)) {
      return {NULL, NULL, NULL, NULL, NULL, NULL};
    }
    pxa::PxaGenericOp pxaAddOp =
        dyn_cast<pxa::PxaGenericOp>(addOp->getDefiningOp());
    if (pxaAddOp.kernel().str() != "tpp_add") {
      return {NULL, NULL, NULL, NULL, NULL, NULL};
    }

    auto gemmOp0Pattern = m_Capture(gemmOp0, m_Op<pxa::PxaGenericOp>());
    if (!matchOperandOrValueAtIndex(pxaAddOp, 0, gemmOp0Pattern)) {
      return {NULL, NULL, NULL, NULL, NULL, NULL};
    }

    pxa::PxaGenericOp pxaGemmOp0 =
        dyn_cast<pxa::PxaGenericOp>(gemmOp0->getDefiningOp());
    if (pxaGemmOp0.kernel().str() != "tpp_gemm") {
      return {NULL, NULL, NULL, NULL, NULL, NULL};
    }
    auto identityOp0Pattern = m_Capture(identityOp0, m_Op<pxa::PxaGenericOp>());
    if (!matchOperandOrValueAtIndex(pxaGemmOp0, 2, identityOp0Pattern)) {
      return {NULL, NULL, NULL, NULL, NULL, NULL};
    }
    pxa::PxaGenericOp pxaIdentityOp0 =
        dyn_cast<pxa::PxaGenericOp>(identityOp0->getDefiningOp());
    if (pxaIdentityOp0.kernel().str() != "tpp_identity") {
      return {NULL, NULL, NULL, NULL, NULL, NULL};
    }

    auto gemmOp1Pattern = m_Capture(gemmOp1, m_Op<pxa::PxaGenericOp>());
    if (!matchOperandOrValueAtIndex(pxaAddOp, 1, gemmOp1Pattern)) {
      return {NULL, NULL, NULL, NULL, NULL, NULL};
    }

    pxa::PxaGenericOp pxaGemmOp1 =
        dyn_cast<pxa::PxaGenericOp>(gemmOp1->getDefiningOp());
    if (pxaGemmOp1.kernel().str() != "tpp_gemm") {
      return {NULL, NULL, NULL, NULL, NULL, NULL};
    }
    auto identityOp1Pattern = m_Capture(identityOp1, m_Op<pxa::PxaGenericOp>());
    if (!matchOperandOrValueAtIndex(pxaGemmOp1, 2, identityOp1Pattern)) {
      return {NULL, NULL, NULL, NULL, NULL, NULL};
    }
    pxa::PxaGenericOp pxaIdentityOp1 =
        dyn_cast<pxa::PxaGenericOp>(identityOp1->getDefiningOp());
    if (pxaIdentityOp1.kernel().str() != "tpp_identity") {
      return {NULL, NULL, NULL, NULL, NULL, NULL};
    }

    return {pxaReluOp,      pxaAddOp,   pxaGemmOp0,
            pxaIdentityOp0, pxaGemmOp1, pxaIdentityOp1};
  }

  SmallVector<pxa::PxaGenericOp, 4>
  maybeCapturePartiallyChainedGemms(AffineParallelOp op) {
    using matchers::m_Any;
    Value *reluOp = new Value();
    Value *gemmOp0 = new Value();
    Value *identityOp0 = new Value();
    Value *addOp = new Value();

    auto opPattern = m_Op<AffineYieldOp>(m_Capture(
        reluOp, m_Op<pxa::PxaGenericOp>())); // m_Capture(gemmOp,
                                             // m_Op<pxa::PxaGenericOp>()))));
    auto affineYield = op.getBody()->getTerminator();
    if (!matchPattern(affineYield, opPattern)) {
      return {NULL, NULL, NULL, NULL};
    }
    pxa::PxaGenericOp pxaReluOp =
        dyn_cast<pxa::PxaGenericOp>(reluOp->getDefiningOp());
    if (pxaReluOp.kernel().str() != "tpp_relu") {
      return {NULL, NULL, NULL, NULL};
    }
    auto addOpPattern = m_Capture(addOp, m_Op<pxa::PxaGenericOp>());
    if (!matchOperandOrValueAtIndex(pxaReluOp, 0, addOpPattern)) {
      return {NULL, NULL, NULL, NULL};
    }
    pxa::PxaGenericOp pxaAddOp =
        dyn_cast<pxa::PxaGenericOp>(addOp->getDefiningOp());
    if (pxaAddOp.kernel().str() != "tpp_add") {
      return {NULL, NULL, NULL, NULL};
    }

    auto gemmOp0Pattern = m_Capture(gemmOp0, m_Op<pxa::PxaGenericOp>());
    if (!matchOperandOrValueAtIndex(pxaAddOp, 0, gemmOp0Pattern) &&
        !matchOperandOrValueAtIndex(pxaAddOp, 1, gemmOp0Pattern)) {
      return {NULL, NULL, NULL, NULL};
    }
    pxa::PxaGenericOp pxaGemmOp0 =
        dyn_cast<pxa::PxaGenericOp>(gemmOp0->getDefiningOp());
    if (pxaGemmOp0.kernel().str() != "tpp_gemm") {
      return {NULL, NULL, NULL, NULL};
    }
    auto identityOp0Pattern = m_Capture(identityOp0, m_Op<pxa::PxaGenericOp>());
    if (!matchOperandOrValueAtIndex(pxaGemmOp0, 2, identityOp0Pattern)) {
      return {NULL, NULL, NULL, NULL};
    }
    pxa::PxaGenericOp pxaIdentityOp0 =
        dyn_cast<pxa::PxaGenericOp>(identityOp0->getDefiningOp());
    if (pxaIdentityOp0.kernel().str() != "tpp_identity") {
      return {NULL, NULL, NULL, NULL};
    }
    return {pxaReluOp, pxaAddOp, pxaGemmOp0, pxaIdentityOp0};
  }
}; // namespace
} // namespace

struct TppCombinePass : public TppCombineBase<TppCombinePass> {
  void runOnOperation() final {
    getOperation().walk([](AffineParallelOp op) {
      TppCombineImpl combineImpl;
      auto opsVector = combineImpl.maybeCaptureGemm(op);
      if (opsVector[0] != NULL && opsVector[1] != NULL &&
          opsVector[2] != NULL) {
        pxa::PxaGenericOp reluOp = opsVector[0];
        pxa::PxaGenericOp gemmOp = opsVector[1];
        pxa::PxaGenericOp identityOp = opsVector[2];
        OpBuilder builder(reluOp);

        SmallVector<Value> inputIndices;
        SmallVector<AffineMap> inputAccessMaps;
        SmallVector<AffineValueMap> inputValueMaps;
        inputValueMaps.reserve(gemmOp.getNumInputs());
        gemmOp.getAffineValueMaps(gemmOp.inputAccessMaps(),
                                  gemmOp.inputIndices(), inputValueMaps);

        for (AffineValueMap &valueMap : inputValueMaps) {
          inputAccessMaps.push_back(valueMap.getAffineMap());
          inputIndices.append(valueMap.getOperands().begin(),
                              valueMap.getOperands().end());
        }
        inputValueMaps.clear();
        identityOp.getAffineValueMaps(identityOp.inputAccessMaps(),
                                      identityOp.inputIndices(),
                                      inputValueMaps);
        for (AffineValueMap &valueMap : inputValueMaps) {
          inputAccessMaps.push_back(valueMap.getAffineMap());
          inputIndices.append(valueMap.getOperands().begin(),
                              valueMap.getOperands().end());
        }
        SmallVector<AffineMap> inputTileMaps;
        for (auto &tileMap : gemmOp.inputTileMaps()) {
          auto tile = tileMap.cast<AffineMapAttr>().getValue();
          inputTileMaps.push_back(tile);
        }

        for (auto &tileMap : identityOp.inputTileMaps()) {
          auto tile = tileMap.cast<AffineMapAttr>().getValue();
          inputTileMaps.push_back(tile);
        }
        auto genericOp = builder.create<pxa::PxaGenericOp>(
            gemmOp.getLoc(), reluOp.getResult(0).getType(),
            /*inputs=*/
            ArrayRef<Value>{gemmOp.getOperand(0), gemmOp.getOperand(1),
                            identityOp.getOperand(0)},
            /*outputs=*/ArrayRef<Value>{reluOp.getOperand(1)},
            /*inputIndices=*/inputIndices,
            /*outputIndices=*/reluOp.outputIndices(),
            /*inputAccessMaps=*/builder.getAffineMapArrayAttr(inputAccessMaps),
            /*inputTileMaps=*/builder.getAffineMapArrayAttr(inputTileMaps),
            /*outputAccessMaps=*/reluOp.outputAccessMaps(),
            /*outputTileMaps=*/reluOp.outputTileMaps(),
            /*kernel=*/builder.getStringAttr("tpp_gemm_relu"),
            /*tile=*/gemmOp.tile(),
            /*reductions=*/
            builder.getI64ArrayAttr(
                {static_cast<int64_t>(arith::AtomicRMWKind::assign)}));
        reluOp.getResult(0).replaceAllUsesWith(genericOp.getResult(0));
        reluOp.erase();
        gemmOp.erase();
        identityOp.erase();
        auto allocOp = identityOp.getOperand(1).getDefiningOp();
        if (cast<memref::AllocOp>(allocOp)) {
          auto deallocOp = pmlc::util::findDeallocPair(allocOp);
          deallocOp->erase();
          allocOp->erase();
        }
      } else {
        auto sixOpsVector = combineImpl.maybeCaptureChainedGemms(op);
        if (sixOpsVector[0] != NULL && sixOpsVector[1] != NULL &&
            sixOpsVector[2] != NULL && sixOpsVector[3] != NULL &&
            sixOpsVector[4] != NULL && sixOpsVector[5] != NULL) {
          pxa::PxaGenericOp reluOp = sixOpsVector[0];
          pxa::PxaGenericOp addOp = sixOpsVector[1];
          pxa::PxaGenericOp gemmOp0 = sixOpsVector[2];
          pxa::PxaGenericOp identityOp0 = sixOpsVector[3];
          pxa::PxaGenericOp gemmOp1 = sixOpsVector[4];
          pxa::PxaGenericOp identityOp1 = sixOpsVector[5];
          OpBuilder builder(reluOp);

          SmallVector<Value> inputIndices0;
          SmallVector<AffineMap> inputAccessMaps0;
          SmallVector<AffineValueMap> inputValueMaps0;
          inputValueMaps0.reserve(gemmOp0.getNumInputs());
          gemmOp0.getAffineValueMaps(gemmOp0.inputAccessMaps(),
                                     gemmOp0.inputIndices(), inputValueMaps0);

          for (AffineValueMap &valueMap : inputValueMaps0) {
            inputAccessMaps0.push_back(valueMap.getAffineMap());
            inputIndices0.append(valueMap.getOperands().begin(),
                                 valueMap.getOperands().end());
          }
          inputValueMaps0.clear();
          identityOp0.getAffineValueMaps(identityOp0.inputAccessMaps(),
                                         identityOp0.inputIndices(),
                                         inputValueMaps0);
          for (AffineValueMap &valueMap : inputValueMaps0) {
            inputAccessMaps0.push_back(valueMap.getAffineMap());
            inputIndices0.append(valueMap.getOperands().begin(),
                                 valueMap.getOperands().end());
          }
          SmallVector<AffineMap> inputTileMaps0;
          for (auto &tileMap : gemmOp0.inputTileMaps()) {
            auto tile = tileMap.cast<AffineMapAttr>().getValue();
            inputTileMaps0.push_back(tile);
          }

          for (auto &tileMap : identityOp0.inputTileMaps()) {
            auto tile = tileMap.cast<AffineMapAttr>().getValue();
            inputTileMaps0.push_back(tile);
          }
          auto genericOp0 = builder.create<pxa::PxaGenericOp>(
              gemmOp0.getLoc(), reluOp.getResult(0).getType(),
              /*inputs=*/
              ArrayRef<Value>{gemmOp0.getOperand(0), gemmOp0.getOperand(1),
                              identityOp0.getOperand(0)},
              /*outputs=*/ArrayRef<Value>{reluOp.getOperand(1)},
              /*inputIndices=*/inputIndices0,
              /*outputIndices=*/reluOp.outputIndices(),
              /*inputAccessMaps=*/
              builder.getAffineMapArrayAttr(inputAccessMaps0),
              /*inputTileMaps=*/builder.getAffineMapArrayAttr(inputTileMaps0),
              /*outputAccessMaps=*/reluOp.outputAccessMaps(),
              /*outputTileMaps=*/reluOp.outputTileMaps(),
              /*kernel=*/builder.getStringAttr("tpp_gemm_bias"),
              /*tile=*/gemmOp0.tile(),
              /*reductions=*/
              builder.getI64ArrayAttr(
                  {static_cast<int64_t>(arith::AtomicRMWKind::assign)}));

          SmallVector<Value> inputIndices1;
          SmallVector<AffineMap> inputAccessMaps1;
          SmallVector<AffineValueMap> inputValueMaps1;
          inputValueMaps1.reserve(gemmOp1.getNumInputs());
          gemmOp1.getAffineValueMaps(gemmOp1.inputAccessMaps(),
                                     gemmOp1.inputIndices(), inputValueMaps1);

          for (AffineValueMap &valueMap : inputValueMaps1) {
            inputAccessMaps1.push_back(valueMap.getAffineMap());
            inputIndices1.append(valueMap.getOperands().begin(),
                                 valueMap.getOperands().end());
          }
          inputValueMaps1.clear();
          identityOp1.getAffineValueMaps(identityOp1.inputAccessMaps(),
                                         identityOp1.inputIndices(),
                                         inputValueMaps1);
          for (AffineValueMap &valueMap : inputValueMaps1) {
            inputAccessMaps1.push_back(valueMap.getAffineMap());
            inputIndices1.append(valueMap.getOperands().begin(),
                                 valueMap.getOperands().end());
          }
          SmallVector<AffineMap> inputTileMaps1;
          for (auto &tileMap : gemmOp1.inputTileMaps()) {
            auto tile = tileMap.cast<AffineMapAttr>().getValue();
            inputTileMaps1.push_back(tile);
          }

          for (auto &tileMap : identityOp1.inputTileMaps()) {
            auto tile = tileMap.cast<AffineMapAttr>().getValue();
            inputTileMaps1.push_back(tile);
          }
          auto genericOp1 = builder.create<pxa::PxaGenericOp>(
              gemmOp1.getLoc(), genericOp0.getResult(0).getType(),
              /*inputs=*/
              ArrayRef<Value>{gemmOp1.getOperand(0), gemmOp1.getOperand(1),
                              identityOp1.getOperand(0)},
              /*outputs=*/ArrayRef<Value>{genericOp0.getResult(0)},
              /*inputIndices=*/inputIndices1,
              /*outputIndices=*/reluOp.outputIndices(),
              /*inputAccessMaps=*/
              builder.getAffineMapArrayAttr(inputAccessMaps1),
              /*inputTileMaps=*/builder.getAffineMapArrayAttr(inputTileMaps1),
              /*outputAccessMaps=*/reluOp.outputAccessMaps(),
              /*outputTileMaps=*/reluOp.outputTileMaps(),
              /*kernel=*/builder.getStringAttr("tpp_gemm_relu_beta1"),
              /*tile=*/gemmOp1.tile(),
              /*reductions=*/
              builder.getI64ArrayAttr(
                  {static_cast<int64_t>(arith::AtomicRMWKind::assign)}));
          reluOp.getResult(0).replaceAllUsesWith(genericOp1.getResult(0));
          reluOp.erase();
          addOp.erase();
          gemmOp0.erase();
          identityOp0.erase();
          gemmOp1.erase();
          identityOp1.erase();
          auto allocOp0 = identityOp0.getOperand(1).getDefiningOp();
          if (cast<memref::AllocOp>(allocOp0)) {
            auto deallocOp0 = pmlc::util::findDeallocPair(allocOp0);
            deallocOp0->erase();
            allocOp0->erase();
          }
          auto allocOp1 = identityOp1.getOperand(1).getDefiningOp();
          if (cast<memref::AllocOp>(allocOp1)) {
            auto deallocOp1 = pmlc::util::findDeallocPair(allocOp1);
            deallocOp1->erase();
            allocOp1->erase();
          }
          auto allocOp2 = addOp.getOperand(2).getDefiningOp();
          if (cast<memref::AllocOp>(allocOp2)) {
            auto deallocOp2 = pmlc::util::findDeallocPair(allocOp2);
            deallocOp2->erase();
            allocOp2->erase();
          }
        } else {
          auto fourOpsVector =
              combineImpl.maybeCapturePartiallyChainedGemms(op);
          if (fourOpsVector[0] != NULL && fourOpsVector[1] != NULL &&
              fourOpsVector[2] != NULL && fourOpsVector[3] != NULL) {
            pxa::PxaGenericOp reluOp = fourOpsVector[0];
            pxa::PxaGenericOp addOp = fourOpsVector[1];
            pxa::PxaGenericOp gemmOp0 = fourOpsVector[2];
            pxa::PxaGenericOp identityOp0 = fourOpsVector[3];
            Value addBuffer;
            if (addOp.getOperand(0).getDefiningOp() == gemmOp0) {
              addBuffer = addOp.getOperand(1);
            } else {
              addBuffer = addOp.getOperand(0);
            }
            Value reluBuffer = addBuffer;
            while (true) {
              Value *prevReluOp = new Value();
              auto opPattern = m_Op<AffineYieldOp>(
                  m_Capture(prevReluOp, m_Op<pxa::PxaGenericOp>()));
              auto prevOp = cast<AffineParallelOp>(reluBuffer.getDefiningOp());
              auto affineYield = prevOp.getBody()->getTerminator();
              assert(matchPattern(affineYield, opPattern));
              pxa::PxaGenericOp prevPxaReluOp =
                  dyn_cast<pxa::PxaGenericOp>(prevReluOp->getDefiningOp());
              if (prevPxaReluOp.kernel() == "tpp_relu") {
                reluBuffer = prevPxaReluOp.getOperand(1);
              } else if (prevPxaReluOp.kernel() == "tpp_gemm_relu" ||
                         prevPxaReluOp.kernel() == "tpp_gemm_relu_beta1" ||
                         prevPxaReluOp.kernel() == "tpp_gemm_bias") {
                reluBuffer = prevPxaReluOp.getOperand(3);
              }
              if (isa<pxa::PxaGenericOp>(reluBuffer.getDefiningOp())) {
                prevPxaReluOp =
                    cast<pxa::PxaGenericOp>(reluBuffer.getDefiningOp());
                if (prevPxaReluOp.kernel() == "tpp_relu") {
                  reluBuffer = prevPxaReluOp.getOperand(1);
                } else if (prevPxaReluOp.kernel() == "tpp_gemm_relu" ||
                           prevPxaReluOp.kernel() == "tpp_gemm_relu_beta1" ||
                           prevPxaReluOp.kernel() == "tpp_gemm_bias") {
                  reluBuffer = prevPxaReluOp.getOperand(3);
                }
              }
              if (isa<memref::AllocOp>(reluBuffer.getDefiningOp())) {
                break;
              }
            }
            OpBuilder builder(reluOp);

            SmallVector<Value> inputIndices0;
            SmallVector<AffineMap> inputAccessMaps0;
            SmallVector<AffineValueMap> inputValueMaps0;
            inputValueMaps0.reserve(gemmOp0.getNumInputs());
            gemmOp0.getAffineValueMaps(gemmOp0.inputAccessMaps(),
                                       gemmOp0.inputIndices(), inputValueMaps0);

            for (AffineValueMap &valueMap : inputValueMaps0) {
              inputAccessMaps0.push_back(valueMap.getAffineMap());
              inputIndices0.append(valueMap.getOperands().begin(),
                                   valueMap.getOperands().end());
            }
            inputValueMaps0.clear();
            identityOp0.getAffineValueMaps(identityOp0.inputAccessMaps(),
                                           identityOp0.inputIndices(),
                                           inputValueMaps0);
            for (AffineValueMap &valueMap : inputValueMaps0) {
              inputAccessMaps0.push_back(valueMap.getAffineMap());
              inputIndices0.append(valueMap.getOperands().begin(),
                                   valueMap.getOperands().end());
            }
            SmallVector<AffineMap> inputTileMaps0;
            for (auto &tileMap : gemmOp0.inputTileMaps()) {
              auto tile = tileMap.cast<AffineMapAttr>().getValue();
              inputTileMaps0.push_back(tile);
            }

            for (auto &tileMap : identityOp0.inputTileMaps()) {
              auto tile = tileMap.cast<AffineMapAttr>().getValue();
              inputTileMaps0.push_back(tile);
            }
            auto genericOp0 = builder.create<pxa::PxaGenericOp>(
                gemmOp0.getLoc(), reluOp.getResult(0).getType(),
                /*inputs=*/
                ArrayRef<Value>{gemmOp0.getOperand(0), gemmOp0.getOperand(1),
                                identityOp0.getOperand(0)},
                /*outputs=*/ArrayRef<Value>{addBuffer},
                /*inputIndices=*/inputIndices0,
                /*outputIndices=*/reluOp.outputIndices(),
                /*inputAccessMaps=*/
                builder.getAffineMapArrayAttr(inputAccessMaps0),
                /*inputTileMaps=*/builder.getAffineMapArrayAttr(inputTileMaps0),
                /*outputAccessMaps=*/reluOp.outputAccessMaps(),
                /*outputTileMaps=*/reluOp.outputTileMaps(),
                /*kernel=*/builder.getStringAttr("tpp_gemm_relu_beta1"),
                /*tile=*/gemmOp0.tile(),
                /*reductions=*/
                builder.getI64ArrayAttr(
                    {static_cast<int64_t>(arith::AtomicRMWKind::assign)}));
            reluOp.getResult(0).replaceAllUsesWith(genericOp0.getResult(0));
            reluOp.erase();
            addOp.erase();
            gemmOp0.erase();
            identityOp0.erase();
            auto allocOp0 = identityOp0.getOperand(1).getDefiningOp();
            if (isa<memref::AllocOp>(allocOp0)) {
              auto deallocOp0 = pmlc::util::findDeallocPair(allocOp0);
              deallocOp0->erase();
              allocOp0->erase();
            }
            auto allocOp1 = addOp.getOperand(2).getDefiningOp();
            if (isa<memref::AllocOp>(allocOp1)) {
              auto deallocOp1 = pmlc::util::findDeallocPair(allocOp1);
              deallocOp1->erase();
              allocOp1->erase();
            }
            auto deallocForReluBuffer =
                pmlc::util::findDeallocPair(reluBuffer.getDefiningOp());
            auto deallocForAddOperand = pmlc::util::findDeallocPair(
                reluOp.getOperand(1).getDefiningOp());
            deallocForAddOperand->replaceUsesOfWith(reluOp.getOperand(1),
                                                    reluBuffer);
            reluOp.getOperand(1).getDefiningOp()->erase();
            deallocForReluBuffer->erase();
          }
        }
      }
    });
  }
};

std::unique_ptr<Pass> createTppCombinePass() {
  return std::make_unique<TppCombinePass>();
}

} // namespace pmlc::target::x86
