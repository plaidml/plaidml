// Copyright 2020 Intel Corporation

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Support/DebugStringHelper.h" // Lubo

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/transforms/vectorize.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::pxa {
using pmlc::dialect::pxa::AffineReduceOp;

class Impl {
  enum OpType {
    NONE,
    LOAD,
    REDUCE,
    SCALAR,
  };

  struct OpVectState {
    Operation *op;  // The operation
    OpType opType;  // Type of the operation
    int64_t stride; // The strides for the index evaluated.

    OpVectState(Operation *op, OpType opType, int64_t stride)
        : op(op), opType(opType), stride(stride) {}
  };

private:
  AffineParallelOp op;
  BlockArgument index;
  unsigned vectorSize;
  unsigned minElemWidth;
  std::unordered_map<Operation *, OpVectState> vectorizableOps;
  unsigned numElementsInRegister;
  VectorType vecType;

  bool tryVectorizeOperation(Operation *op) {
    bool ret = true;
    llvm::TypeSwitch<Operation *>(op)
        .Case<AffineLoadOp>([&](auto op) {
          auto strideInfo = computeStrideInfo(op);
          if (!strideInfo) {
            ret = false;
            return;
          }

          auto pair = strideInfo->strides.find(index);
          if (pair == strideInfo->strides.end()) {
            IVLOG(1, "Vectorize: Lubo: AffineLoadOp stride of 0 for index");
            // stride of 0.
            OpVectState opState(op, OpType::LOAD, 0);
            vectorizableOps.emplace(op, opState);
            return;
          }

          if (pair->second != 1) {
            IVLOG(1, "Cannot vectorize: Lubo AffineLoadOp stride different "
                     "than 0 and 1 for index:"
                         << pair->second);
            ret = false;
            return;
          }
          OpVectState opState(op, OpType::LOAD, 1);
          vectorizableOps.emplace(op, opState);
          ret = true;
          return;
        })
        .Case<AffineReduceOp>([&](auto op) {
          auto strideInfo = computeStrideInfo(op);
          if (!strideInfo) {
            ret = false;
            return;
          }

          auto pair = strideInfo->strides.find(index);
          if (pair == strideInfo->strides.end()) {
            IVLOG(1, "Vectorize: Lubo: AffineReduceOp stride of 0 for index  "
                     "no stride info");
            // TODO: Deal with reductions of stride 0.
            // TODO: Call vector_reduce and then scalar pxa.reduce (to have the
            // MemRefState properly returned).
            ret = false;
            return;
          }

          if (pair->second != 1) {
            IVLOG(1, "Cannot vectorize: AffineReduceOp stride different than 0 "
                     "and 1 for index");
            ret = false;
            return;
          }

          OpVectState opState(op, OpType::REDUCE, 1);
          vectorizableOps.emplace(op, opState);
          return;
        })
        .Default([&](Operation *op) {
          // Lubo auto casted = mlir::cast<VectorUnrollOpInterface>(op);
          if (mlir::isa<VectorUnrollOpInterface>(op)) {
            OpVectState opState(op, OpType::SCALAR, -1);
            vectorizableOps.emplace(op, opState);
            return;
          }
        });
    return ret;
  }

  // Get the element width in bytes (8 bits).
  unsigned getElementWidth(AffineParallelOp forOp) {
    unsigned width = minElemWidth;
    for (auto &op : forOp.getLoopBody().front()) {
      if (auto load = mlir::dyn_cast<AffineLoadOp>(op)) {
        unsigned w = load.getResult().getType().getIntOrFloatBitWidth() / 8;
        width = std::max(w, width);
      }
    }
    return width;
  }

  // Get the element type.
  Type getElementType(AffineParallelOp forOp) {
    for (auto &op : forOp.getLoopBody().front()) {
      if (auto load = mlir::dyn_cast<AffineLoadOp>(op)) {
        return load.getResult().getType();
      }
    }
    llvm_unreachable("Vectorize: Nlo load operation to get element type");
  }

public:
  Impl(AffineParallelOp op, BlockArgument index, unsigned vectorSize,
       unsigned minElemWidth)
      : op(op), index(index), vectorSize(vectorSize),
        minElemWidth(minElemWidth), numElementsInRegister(0) {}

  Operation *vectorizeOperation(OpBuilder &builder, Operation &loopOperation) {
    auto pair = vectorizableOps.find(&loopOperation);
    if (pair != vectorizableOps.end()) {
      OpVectState opVectState = pair->second;
      IVLOG(1, "Lubo: Found operation:" << opVectState.opType << ":"
                                        << opVectState.stride);

      // TODO: transform

      if (opVectState.opType == OpType::SCALAR) {
        Operation *ret = builder.clone(loopOperation);
        // TODO: ret->setType(vecType);
        return ret;
      }
    }

    return builder.clone(loopOperation);
  }

  void createAndPopulateNewLoop(AffineParallelOp op, unsigned argNum) {
    mlir::Block *block = op.getOperation()->getBlock();
    // mlir::OpBuilder builder(op); // Lubo , op.getBody()->begin());
    IVLOG(1, "Lubo10: " << mlir::debugString(*op.getParentOp()->getParentOp()));
    // Lubo Operation* luboOp = op.getParentOp()->getParentOp();
    mlir::OpBuilder builder(op);
    // Fix up the steps
    std::vector<int64_t> newSteps;
    for (unsigned int i = 0; i < op.getIVs().size(); i++) {
      newSteps.push_back(op.steps().getValue()[i].cast<IntegerAttr>().getInt() *
                         (i == argNum ? numElementsInRegister : 1));
    }
    SmallVector<AtomicRMWKind, 8> reductions(op.getResultTypes().size(),
                                             AtomicRMWKind::assign);
    AffineParallelOp newAffineParallelOp = builder.create<AffineParallelOp>(
        op.getLoc(), op.getResultTypes(), reductions,
        op.getLowerBoundsValueMap().getAffineMap(), op.getLowerBoundsOperands(),
        op.getUpperBoundsValueMap().getAffineMap(), op.getUpperBoundsOperands(),
        newSteps);
    IVLOG(1,
          "Lubo8: " << mlir::debugString(*newAffineParallelOp.getOperation()));
    for (auto it = block->begin(); it != block->end(); it++) {
      // Operation* blockOp = &*it;
      IVLOG(1, "Lubo4: " << mlir::debugString(*it));
      for (unsigned i = 0; i < (*it).getNumOperands(); i++) {
        Value operand = (*it).getOperand(i);
        if (operand.getDefiningOp() == op) {
          IVLOG(1, "Lubo5: " << i);
          // builder.insert(newAffineParallelOp);
          (*it).setOperand(i, newAffineParallelOp.getResult(0));

          // Now transfer the body
          mlir::Block *newLoopBlock = newAffineParallelOp.getBody();
          // Lubo OpBuilder newBlockBuilder = OpBuilder(&newLoopBlock);
          builder.setInsertionPointToStart(newLoopBlock);

          // Walk over the statements in order.
          for (auto &loopOp : llvm::make_early_inc_range(*op.getBody())) {
            // IVLOG(1, "Lubo6:" << mlir::debugString(loopOp));
            /* The prtinting is broken with the latest project-llvm sources
             * Operation* clonedOp =*/
            vectorizeOperation(builder, loopOp);
            /*Operation clonedOp = builder.clone(loopOp); */
            // IVLOG(1, "Lubo7:" << mlir::debugString(*clonedOp));
            // IVLOG(1, "Lubo17:" << mlir::debugString(loopOp));
            // loopOp.dropAllUses();
            // loopOp.erase();
          }

          // Now erase the instructions
          for (auto &loopOp : llvm::make_early_inc_range(*op.getBody())) {
            // IVLOG(1, "Lubo6:" << mlir::debugString(loopOp));
            // / * Operation* clonedOp =* / builder.clone(loopOp);
            // IVLOG(1, "Lubo7:" << mlir::debugString(*clonedOp));
            // IVLOG(1, "Lubo17:" << mlir::debugString(loopOp));
            loopOp.dropAllUses();
            loopOp.dropAllReferences();
            loopOp.erase();
          }
          // IVLOG(1, "Lubo9: " <<
          // mlir::debugString(*newAffineParallelOp.getOperation()));

          IVLOG(1, "Lubo12: " << mlir::debugString(*op.getOperation()));
        }
      }
    }

    IVLOG(1, "Lubo10.2: ");
    // IVLOG(1, "Lubo10.1: " <<
    // mlir::debugString(*op.getParentOp()->getParentOp()));
    IVLOG(1, "Lubo10.3: ");
    // Remove the original ForOp
    mlir::Region *parentRegion = op.getOperation()->getParentRegion();
    if (!parentRegion) {
      llvm_unreachable("AffineParallelOp with no parent region.");
    }

    // IVLOG(1, "Lubo14: " << mlir::debugString(*luboOp));

    auto &opList = parentRegion->getBlocks().front().getOperations();
    for (auto elt = opList.begin(); elt != opList.end(); ++elt) {
      if (&(*elt) == op.getOperation()) {
        IVLOG(1, "Lubo13: " << mlir::debugString(*elt) << ":"
                            << (*elt).getResults().size());
        IVLOG(1, "Lubo15: " << elt->use_empty());
        for (auto itU = elt->use_begin(); itU != elt->use_end(); itU++) {
          IVLOG(1,
                "Lubo16: " << mlir::debugString(*itU->get().getDefiningOp()));
        }
        IVLOG(1, "Lubo18: " << mlir::debugString(*elt));
        // elt->dropAllUses();
        // elt->dropAllReferences();
        opList.erase(elt);
        break;
      }
    }
    // IVLOG(1, "Lubo11: " << mlir::debugString(*luboOp));
  }

  bool vectorize() {
    mlir::Block *outerBody = op.getBody();
    IVLOG(1, "Lubo: Vectorizing for BlockArgument: " << index.getArgNumber());
    unsigned elementWidth = getElementWidth(op);
    if ((vectorSize % elementWidth) != 0) {
      IVLOG(1, "Cannot Vectorize: The vector size is not a multiple of the "
               "element type size");
      return false;
    }

    auto ranges = op.getConstantRanges();
    if (!ranges) {
      IVLOG(1, "Cannot Vectorize: Requires constant ranges");
      return false;
    }

    numElementsInRegister = vectorSize / elementWidth;

    IVLOG(1, "Lubo30:" << numElementsInRegister << ":" << vectorSize << ":"
                       << elementWidth);

    vecType = VectorType::get(ArrayRef<int64_t>{numElementsInRegister},
                              getElementType(op)); // TODO: Use shape...

    auto argNum = index.getArgNumber();
    if (((*ranges)[argNum] % numElementsInRegister) > 0) {
      if ((*ranges)[argNum] < 0) {
        IVLOG(1, "Cannot Vectorize: number of elements less than the register "
                 "elements");
        return false;
      }
      // TODO: Can we do partial use of vectors here?
      IVLOG(1, "Cannot Vectorize: The dimension being vectorized not multiple "
               "of the number of elements in a register");
      return false;
    }

    auto step = op.steps().getValue()[argNum].cast<IntegerAttr>().getInt();
    if (step != 1) {
      IVLOG(1, "Cannot Vectorize: The steps for the dimension being vectorized "
               "is not 1");
      return false;
    }

    bool vectorizable = true;
    outerBody->walk(
        [&](Operation *op) { vectorizable &= tryVectorizeOperation(op); });
    // In case no operations were marked for vectorizations.
    if (vectorizableOps.size() == 0) {
      vectorizable = false;
    }

    if (!vectorizable) {
      IVLOG(1, "Cannot Vectorize: No instructions selected for vectorization");
      return false;
    }

    IVLOG(1, "Vectorize: Lubo: Operations to vectorize:"
                 << vectorizableOps.size() << " for index " << argNum);

    createAndPopulateNewLoop(op, argNum);
    // Create a new
    // Lubo getParentOp()->erase();

    // for (OpVectState vectState : vectorizableOps) {
    //   switch (vectState.opType) {
    //     case OpType::LOAD: {
    //       mlir::OpBuilder builder(vectState.op);
    //         if (vectState.stride == 0) {

    //         } else if (vectState.stride == 1) {

    //         }
    //       }
    //       break;
    //     case OpType::REDUCE:
    //       break;
    //     case OpType::SCALAR:
    //       break;
    //     default:
    //       llvm_unreachable("Invalid Op-Type for vectorization");
    //   }
    // }

    return true;
  }
};

bool performVectorization(AffineParallelOp op, BlockArgument index,
                          unsigned vectorSize, unsigned minElemWidth) {
  Impl impl(op, index, vectorSize, minElemWidth);
  return impl.vectorize();
}
} // namespace pmlc::dialect::pxa
