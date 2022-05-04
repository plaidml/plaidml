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
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir; // NOLINT

namespace pmlc::target::x86 {

namespace pxa = dialect::pxa;
namespace stdx = dialect::stdx;

namespace {

class StencilSplitImpl {
private:
  std::list<AffineParallelOp> transform(SmallVector<Value *, 3> operands,
                                        AffineParallelOp op) {
    std::list<AffineParallelOp> splitOps;
    if (operands[0] == NULL)
      return splitOps;
    Value *reduce = operands[0];
    Value *first = operands[1];
    Value *second = operands[2];

    std::list<Operation *> capturedFirst;
    if (first->getDefiningOp()->getBlock() == reduce->getParentBlock()) {
      capturedFirst.push_back(first->getDefiningOp());
    }
    std::list<Operation *> rootedInstructions;
    while (!capturedFirst.empty()) {
      auto capturedVal = capturedFirst.front();
      capturedFirst.pop_front();
      if (find(rootedInstructions.begin(), rootedInstructions.end(),
               capturedVal) != rootedInstructions.end()) {
        rootedInstructions.erase(find(rootedInstructions.begin(),
                                      rootedInstructions.end(), capturedVal));
      }

      rootedInstructions.push_front(capturedVal);
      for (int i = 0; i < capturedVal->getNumOperands(); i++) {
        if (capturedVal->getOperand(i).getDefiningOp() != NULL &&
            capturedVal->getOperand(i).getDefiningOp()->getBlock() ==
                reduce->getParentBlock()) {
          capturedFirst.push_back(capturedVal->getOperand(i).getDefiningOp());
        }
      }
      if (isa<AffineParallelOp>(capturedVal)) {
        std::list<AffineParallelOp> nestedParallelOps;
        nestedParallelOps.push_back(cast<AffineParallelOp>(capturedVal));
        while (!nestedParallelOps.empty()) {
          auto nestedOp = nestedParallelOps.front();
          nestedParallelOps.pop_front();
          auto body = cast<AffineParallelOp>(nestedOp).getBody();
          for (auto instr = body->begin(); instr != body->end(); instr++) {
            if (isa<AffineParallelOp>(instr)) {
              nestedParallelOps.push_back(cast<AffineParallelOp>(instr));
            }
            for (int i = 0; i < instr->getNumOperands(); i++) {
              if (instr->getOperand(i).getDefiningOp() != NULL &&
                  instr->getOperand(i).getDefiningOp()->getBlock() ==
                      reduce->getParentBlock()) {
                capturedFirst.push_back(instr->getOperand(i).getDefiningOp());
              }
            }
          }
        }
      }
    }
    std::list<Operation *> secondRootedInstructions;
    if (second != NULL) {
      if (second->getDefiningOp()->getBlock() == reduce->getParentBlock()) {
        capturedFirst.push_back(second->getDefiningOp());
      }
      while (!capturedFirst.empty()) {
        auto capturedVal = capturedFirst.front();
        capturedFirst.pop_front();
        if (find(secondRootedInstructions.begin(),
                 secondRootedInstructions.end(),
                 capturedVal) != secondRootedInstructions.end()) {
          secondRootedInstructions.erase(find(secondRootedInstructions.begin(),
                                              secondRootedInstructions.end(),
                                              capturedVal));
        }
        secondRootedInstructions.push_front(capturedVal);
        for (int i = 0; i < capturedVal->getNumOperands(); i++) {
          if (capturedVal->getOperand(i).getDefiningOp() != NULL &&
              capturedVal->getOperand(i).getDefiningOp()->getBlock() ==
                  reduce->getParentBlock()) {
            capturedFirst.push_back(capturedVal->getOperand(i).getDefiningOp());
          }
        }
        if (isa<AffineParallelOp>(capturedVal)) {
          std::list<AffineParallelOp> nestedParallelOps;
          nestedParallelOps.push_back(cast<AffineParallelOp>(capturedVal));
          while (!nestedParallelOps.empty()) {
            auto nestedOp = nestedParallelOps.front();
            nestedParallelOps.pop_front();
            auto body = cast<AffineParallelOp>(nestedOp).getBody();
            for (auto instr = body->begin(); instr != body->end(); instr++) {
              if (isa<AffineParallelOp>(instr)) {
                nestedParallelOps.push_back(cast<AffineParallelOp>(instr));
              }
              for (int i = 0; i < instr->getNumOperands(); i++) {
                if (instr->getOperand(i).getDefiningOp() != NULL &&
                    instr->getOperand(i).getDefiningOp()->getBlock() ==
                        reduce->getParentBlock()) {
                  capturedFirst.push_back(instr->getOperand(i).getDefiningOp());
                }
              }
            }
          }
        }
      }
    }
    std::list<Operation *> thirdRootedInstructions;
    capturedFirst.push_back(reduce->getDefiningOp());
    while (!capturedFirst.empty()) {
      auto capturedVal = capturedFirst.front();
      capturedFirst.pop_front();
      if (capturedVal != first->getDefiningOp() &&
          (second == NULL || capturedVal != second->getDefiningOp()) &&
          capturedVal->getBlock() == reduce->getParentBlock()) {
        if (find(thirdRootedInstructions.begin(), thirdRootedInstructions.end(),
                 capturedVal) != thirdRootedInstructions.end()) {
          thirdRootedInstructions.erase(find(thirdRootedInstructions.begin(),
                                             thirdRootedInstructions.end(),
                                             capturedVal));
        }

        thirdRootedInstructions.push_front(capturedVal);
        for (int i = 0; i < capturedVal->getNumOperands(); i++) {
          if (capturedVal->getOperand(i).getDefiningOp() != NULL) {
            capturedFirst.push_back(capturedVal->getOperand(i).getDefiningOp());
          }
        }
      }
    }
    // There's no need to split the instruction set in this case, as we're
    // looking at a <load,load,reduce> pattern
    if ((rootedInstructions.empty() ||
         (rootedInstructions.size() == 1 &&
          isa<pxa::PxaLoadOp>(rootedInstructions.front()))) &&
        ((secondRootedInstructions.size() == 1 &&
          isa<pxa::PxaLoadOp>(secondRootedInstructions.front())) ||
         secondRootedInstructions.empty())) {
      return splitOps;
    }
    auto loc = op.getBody()->getParentOp()->getParentOp()->getLoc();
    OpBuilder builder(op);
    auto tensorType =
        cast<pxa::PxaReduceOp>(thirdRootedInstructions.back()).getMemRefType();
    auto memRefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    SmallVector<Type, 2> reductionTypes;
    reductionTypes.push_back(memRefType);

    SmallVector<AffineMap, 6> lbMap, ubMap;
    for (int i = 0; i < op.lowerBoundsMap().getNumResults(); i++) {
      lbMap.push_back(op.getLowerBoundMap(i));
    }
    for (int i = 0; i < op.upperBoundsMap().getNumResults(); i++) {
      ubMap.push_back(op.getUpperBoundMap(i));
    }

    auto lbOps = op.getLowerBoundsOperands();
    auto ubOps = op.getUpperBoundsOperands();
    auto steps = op.getSteps();

    AffineMap idMap =
        reduce->getDefiningOp()->getAttrOfType<AffineMapAttr>("map").getValue();
    auto oldReduceOp = cast<pxa::PxaReduceOp>(reduce->getDefiningOp());
    auto range = oldReduceOp.getIdxs();

    if (rootedInstructions.size() == 1 &&
        isa<pxa::PxaLoadOp>(rootedInstructions.front())) {
      thirdRootedInstructions.push_front(rootedInstructions.front());
    }
    if (secondRootedInstructions.size() == 1 &&
        isa<pxa::PxaLoadOp>(secondRootedInstructions.front())) {
      thirdRootedInstructions.push_front(secondRootedInstructions.front());
    }

    mlir::Value newMemFirst;
    AffineParallelOp firstAffineParallelOp;
    if (rootedInstructions.size() > 1 ||
        (rootedInstructions.size() == 1 &&
         !isa<pxa::PxaLoadOp>(rootedInstructions.front()))) {
      newMemFirst = builder.create<mlir::memref::AllocOp>(loc, memRefType);
      firstAffineParallelOp = builder.create<AffineParallelOp>(
          loc, reductionTypes, AtomicRMWKind::assign, lbMap, lbOps, ubMap,
          ubOps, steps);
      auto bodyBuilder =
          OpBuilder::atBlockBegin(firstAffineParallelOp.getBody());
      Operation *lastInstr;
      std::map<Operation *, Operation *> origToCloneInstrMap;
      std::list<Operation *> clonedInstr;
      for (auto firstBlockInstr : rootedInstructions) {
        lastInstr = (bodyBuilder.clone(*firstBlockInstr));
        clonedInstr.push_back(lastInstr);
        origToCloneInstrMap[firstBlockInstr] = lastInstr;
        if (isa<AffineParallelOp>(lastInstr)) {
          std::list<AffineParallelOp> nestedParallelOps;
          nestedParallelOps.push_back(cast<AffineParallelOp>(lastInstr));
          while (!nestedParallelOps.empty()) {
            auto nestedOp = nestedParallelOps.front();
            nestedParallelOps.pop_front();
            auto body = cast<AffineParallelOp>(nestedOp).getBody();
            for (auto instr = body->begin(); instr != body->end(); instr++) {
              if (isa<AffineParallelOp>(instr)) {
                nestedParallelOps.push_back(cast<AffineParallelOp>(instr));
              }
              clonedInstr.push_back(&*instr);
            }
          }
        }
      }
      for (auto clonedInstrItr : clonedInstr) {
        for (auto firstBlockInstr : rootedInstructions) {
          for (int i = 0; i < firstBlockInstr->getNumResults(); i++) {
            clonedInstrItr->replaceUsesOfWith(
                firstBlockInstr->getResult(i),
                origToCloneInstrMap[firstBlockInstr]->getResult(i));
          }
        }
      }

      auto reduceOp = bodyBuilder.create<pxa::PxaReduceOp>(
          firstAffineParallelOp.getLoc(), AtomicRMWKind::assign,
          lastInstr->getResult(0), newMemFirst, idMap, range);
      clonedInstr.push_back(reduceOp);

      for (auto clonedInstrItr : clonedInstr) {
        for (int i = 0; i < clonedInstrItr->getNumOperands(); i++) {
          Value operand = clonedInstrItr->getOperand(i);
          if (std::find(op.getIVs().begin(), op.getIVs().end(), operand) !=
              op.getIVs().end()) {
            clonedInstrItr->replaceUsesOfWith(
                operand, firstAffineParallelOp
                             .getIVs()[std::find(op.getIVs().begin(),
                                                 op.getIVs().end(), operand) -
                                       op.getIVs().begin()]);
          }
        }
      }

      bodyBuilder.create<AffineYieldOp>(firstAffineParallelOp.getLoc(),
                                        ValueRange{reduceOp->getResult(0)});

      splitOps.push_back(firstAffineParallelOp);
    }

    mlir::Value newMemSecond;
    AffineParallelOp secondAffineParallelOp;
    if (secondRootedInstructions.size() > 1 ||
        (secondRootedInstructions.size() == 1 &&
         !isa<pxa::PxaLoadOp>(secondRootedInstructions.front()))) {
      newMemSecond = builder.create<mlir::memref::AllocOp>(loc, memRefType);
      secondAffineParallelOp = builder.create<AffineParallelOp>(
          loc, reductionTypes, AtomicRMWKind::assign, lbMap, lbOps, ubMap,
          ubOps, steps);

      auto bodyBuilder =
          OpBuilder::atBlockBegin(secondAffineParallelOp.getBody());
      Operation *lastInstr;
      std::map<Operation *, Operation *> origToCloneInstrMap;
      std::list<Operation *> clonedInstr;
      for (auto firstBlockInstr : secondRootedInstructions) {
        lastInstr = (bodyBuilder.clone(*firstBlockInstr));
        origToCloneInstrMap[firstBlockInstr] = lastInstr;
        clonedInstr.push_back(lastInstr);
        if (isa<AffineParallelOp>(lastInstr)) {
          std::list<AffineParallelOp> nestedParallelOps;
          nestedParallelOps.push_back(cast<AffineParallelOp>(lastInstr));
          while (!nestedParallelOps.empty()) {
            auto nestedOp = nestedParallelOps.front();
            nestedParallelOps.pop_front();
            auto body = cast<AffineParallelOp>(nestedOp).getBody();
            for (auto instr = body->begin(); instr != body->end(); instr++) {
              if (isa<AffineParallelOp>(instr)) {
                nestedParallelOps.push_back(cast<AffineParallelOp>(instr));
              }
              clonedInstr.push_back(&*instr);
            }
          }
        }
      }
      for (auto clonedInstrItr : clonedInstr) {
        for (auto firstBlockInstr : secondRootedInstructions) {
          for (int i = 0; i < firstBlockInstr->getNumResults(); i++) {
            clonedInstrItr->replaceUsesOfWith(
                firstBlockInstr->getResult(i),
                origToCloneInstrMap[firstBlockInstr]->getResult(i));
          }
        }
      }
      auto reduceOp = bodyBuilder.create<pxa::PxaReduceOp>(
          secondAffineParallelOp.getLoc(), AtomicRMWKind::assign,
          lastInstr->getResult(0), newMemSecond, idMap, range);

      clonedInstr.push_back(reduceOp);
      for (auto clonedInstrItr : clonedInstr) {
        for (int i = 0; i < clonedInstrItr->getNumOperands(); i++) {
          Value operand = clonedInstrItr->getOperand(i);
          if (std::find(op.getIVs().begin(), op.getIVs().end(), operand) !=
              op.getIVs().end()) {
            clonedInstrItr->replaceUsesOfWith(
                operand, secondAffineParallelOp
                             .getIVs()[std::find(op.getIVs().begin(),
                                                 op.getIVs().end(), operand) -
                                       op.getIVs().begin()]);
          }
        }
      }

      bodyBuilder.create<AffineYieldOp>(secondAffineParallelOp.getLoc(),
                                        ValueRange{reduceOp->getResult(0)});
      splitOps.push_back(secondAffineParallelOp);
    }

    {
      AffineParallelOp reduceAffineParallelOp =
          builder.create<AffineParallelOp>(loc, reductionTypes,
                                           AtomicRMWKind::assign, lbMap, lbOps,
                                           ubMap, ubOps, steps);

      auto bodyBuilder =
          OpBuilder::atBlockBegin(reduceAffineParallelOp.getBody());

      Operation *firstLoadMem = NULL;
      if (rootedInstructions.size() > 1 ||
          (rootedInstructions.size() == 1 &&
           !isa<pxa::PxaLoadOp>(rootedInstructions.front()))) {
        firstLoadMem = bodyBuilder.create<pxa::PxaLoadOp>(
            loc, firstAffineParallelOp->getResult(0), idMap, range);
      }
      Operation *secondLoadMem = NULL;
      if (secondRootedInstructions.size() > 1 ||
          (secondRootedInstructions.size() == 1 &&
           !isa<pxa::PxaLoadOp>(secondRootedInstructions.front()))) {
        secondLoadMem = bodyBuilder.create<pxa::PxaLoadOp>(
            loc, secondAffineParallelOp->getResult(0), idMap, range);
      }
      Operation *lastInstr;
      std::map<Operation *, Operation *> origToCloneInstrMap;
      std::list<Operation *> clonedInstr;
      for (auto firstBlockInstr : thirdRootedInstructions) {
        lastInstr = (bodyBuilder.clone(*firstBlockInstr));
        origToCloneInstrMap[firstBlockInstr] = lastInstr;
        clonedInstr.push_back(lastInstr);
      }
      for (auto firstBlockInstr : thirdRootedInstructions) {
        if (firstBlockInstr ==
            reduce->getDefiningOp()->getOperand(0).getDefiningOp()) {
          auto reductionInstr = origToCloneInstrMap[firstBlockInstr];
          if (firstLoadMem != NULL) {
            reductionInstr->replaceUsesOfWith(reductionInstr->getOperand(0),
                                              firstLoadMem->getResult(0));
          }
          if (secondLoadMem != NULL) {
            reductionInstr->replaceUsesOfWith(reductionInstr->getOperand(1),
                                              secondLoadMem->getResult(0));
          }
        }
        for (int i = 0; i < firstBlockInstr->getNumResults(); i++) {
          for (auto clonedInstrItr : clonedInstr)
            clonedInstrItr->replaceUsesOfWith(
                firstBlockInstr->getResult(i),
                origToCloneInstrMap[firstBlockInstr]->getResult(i));
        }
      }
      if (firstLoadMem) {
        clonedInstr.push_back(firstLoadMem);
      }
      if (secondLoadMem) {
        clonedInstr.push_back(secondLoadMem);
      }
      for (auto clonedInstrItr : clonedInstr) {
        for (int i = 0; i < clonedInstrItr->getNumOperands(); i++) {
          Block *parentBlock = clonedInstrItr->getBlock();
          // Cloned instruction uses induction variable as an argument
          Value operand = clonedInstrItr->getOperand(i);
          if (std::find(op.getIVs().begin(), op.getIVs().end(), operand) !=
              op.getIVs().end()) {
            clonedInstrItr->replaceUsesOfWith(
                operand, reduceAffineParallelOp
                             .getIVs()[std::find(op.getIVs().begin(),
                                                 op.getIVs().end(), operand) -
                                       op.getIVs().begin()]);
          }
        }
      }

      bodyBuilder.create<AffineYieldOp>(reduceAffineParallelOp.getLoc(),
                                        ValueRange{lastInstr->getResult(0)});
      op->getResult(0).replaceAllUsesWith(reduceAffineParallelOp.getResult(0));
      splitOps.push_back(reduceAffineParallelOp);
    }
    if (rootedInstructions.size() > 1) {
      builder.create<mlir::memref::DeallocOp>(loc, newMemFirst);
    }
    if (secondRootedInstructions.size() > 1) {
      builder.create<mlir::memref::DeallocOp>(loc, newMemSecond);
    }
    op->dropAllReferences();
    op->dropAllDefinedValueUses();
    op->erase();
    return splitOps;
  }
  template <typename OpTy0>
  SmallVector<Value *, 3> maybeCaptureTopLevel(bool matchBinaryPattern,
                                               AffineParallelOp op) {
    using matchers::m_Any;
    // Check for unary /binary ops back to back within the affineparallel op

    Value *reduce = new Value();
    Value *firstOp = new Value();
    Value *secondOp = new Value();

    auto binaryPattern = m_Op<AffineYieldOp>(m_Capture(
        reduce,
        pxa::m_PxaReduceOp(AtomicRMWKind::assign,
                           m_Op<OpTy0>(m_Capture(firstOp), m_Capture(secondOp)),
                           m_Any())));

    auto affineYield = op.getBody()->getTerminator();
    if (matchBinaryPattern) {
      if (!matchPattern(affineYield, binaryPattern)) {
        return {NULL, NULL, NULL};
      }
      return {reduce, firstOp, secondOp};
    }

    auto unaryPattern = m_Op<AffineYieldOp>(m_Capture(
        reduce, pxa::m_PxaReduceOp(AtomicRMWKind::assign,
                                   m_Op<OpTy0>(m_Capture(firstOp)), m_Any())));

    if (!matchPattern(affineYield, unaryPattern)) {
      return {NULL, NULL, NULL};
    }
    return {reduce, firstOp, NULL};
  }

  SmallVector<Value *, 3> captureTopLevel(AffineParallelOp op) {
    auto retVal = maybeCaptureTopLevel<arith::AddFOp>(true, op);
    if (retVal[0] == NULL) {
      retVal = maybeCaptureTopLevel<arith::MulFOp>(true, op);
    }
    if (retVal[0] == NULL) {
      retVal = maybeCaptureTopLevel<arith::SubFOp>(true, op);
    }
    if (retVal[0] == NULL) {
      retVal = maybeCaptureTopLevel<arith::DivFOp>(true, op);
    }
    if (retVal[0] == NULL) {
      retVal = maybeCaptureTopLevel<stdx::ReluOp>(false, op);
    }
    if (retVal[0] == NULL) {
      retVal = maybeCaptureTopLevel<math::TanhOp>(false, op);
    }
    if (retVal[0] == NULL) {
      retVal = maybeCaptureTopLevel<math::ExpOp>(false, op);
    }
    return retVal;
  }

public:
  void performSplit(AffineParallelOp op) {
    std::list<AffineParallelOp> traversalList;
    traversalList.push_back(op);
    while (!traversalList.empty()) {
      AffineParallelOp op = traversalList.front();
      traversalList.pop_front();
      auto retVal = captureTopLevel(op);
      auto splitOps = transform(retVal, op);
      for (auto splitOp : splitOps) {
        traversalList.push_front(splitOp);
      }
    }
  }
};
} // namespace

struct StencilSplitPass : public StencilSplitBase<StencilSplitPass> {
  void runOnFunction() final {
    getFunction().walk([](AffineParallelOp op) {
      StencilSplitImpl splitImpl;
      splitImpl.performSplit(op);
    });
  }
};

std::unique_ptr<Pass> createStencilSplitPass() {
  return std::make_unique<StencilSplitPass>();
}

} // namespace pmlc::target::x86
