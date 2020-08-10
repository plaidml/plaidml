// Copyright 2020 Intel Corporation

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Vector/VectorOps.h"
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

  void mapResults(Operation &orig, Operation *newOp) {
    IVLOG(1, "Lubo: Mapped operation results1: " << &orig << ":" << newOp);
    IVLOG(1, "Lubo: Mapped operation results1: " << mlir::debugString(orig)
                                                 << ":"
                                                 << mlir::debugString(*newOp));
    orig.replaceAllUsesWith(newOp);
  }

  template <class T>
  Operation *createCmpOp(T op, OpBuilder &builder) {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    return builder.create<T>(op.getLoc(), vecType, op.getPredicate(), lhs, rhs);
  }

  template <class T>
  Operation *createScalarOp(T op, OpBuilder &builder) {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    Operation *ret = builder.create<T>(op.getLoc(), vecType, lhs, rhs);
    IVLOG(1, "Lubo: New SCALAR op:" << mlir::debugString(vecType) << ":"
                                    << mlir::debugString(*ret));
    return ret;
  }

  Operation *vectorizeScalarOp(Operation *op, OpBuilder &builder) {
    Operation *ret;
    TypeSwitch<Operation *>(op)
        .Case<AddFOp>(
            [&](auto op) { ret = createScalarOp<AddFOp>(op, builder); })
        .Case<AddIOp>(
            [&](auto op) { ret = createScalarOp<AddIOp>(op, builder); })
        .Case<SubFOp>(
            [&](auto op) { ret = createScalarOp<SubFOp>(op, builder); })
        .Case<SubIOp>(
            [&](auto op) { ret = createScalarOp<SubIOp>(op, builder); })
        .Case<MulFOp>(
            [&](auto op) { ret = createScalarOp<MulFOp>(op, builder); })
        .Case<MulIOp>(
            [&](auto op) { ret = createScalarOp<MulIOp>(op, builder); })
        .Case<DivFOp>(
            [&](auto op) { ret = createScalarOp<DivFOp>(op, builder); })
        .Case<CmpFOp>([&](auto op) { ret = createCmpOp<CmpFOp>(op, builder); })
        .Case<CmpIOp>([&](auto op) { ret = createCmpOp<CmpIOp>(op, builder); })
        .Default([&](Operation *op) { ret = builder.clone(*op); });
    return ret;
  }

  Operation *vectorizeLoadOp(Operation *op, OpVectState &opVectState,
                             OpBuilder &builder) {
    IVLOG(1, "Lubo: New LOAD op111:" << mlir::debugString(vecType) << ":"
                                     << mlir::debugString(*op));
    AffineLoadOp loadOp = dyn_cast_or_null<AffineLoadOp>(op);
    if (!loadOp) {
      llvm_unreachable("Vectorize: Invalid state for AffineLoadOp");
    }

    Value operand = loadOp.getMemRef();
    Type tp = operand.getType(); // Lubo
    IVLOG(1, "Lubo: New LOAD op222:" << mlir::debugString(tp));

    Operation *ret = nullptr;
    if (opVectState.stride == 0) {
      Operation *cloned = builder.clone(*op);
      ret = builder.create<vector::BroadcastOp>(op->getLoc(), vecType,
                                                cloned->getResults()[0]);
    } else if (opVectState.stride == 1) {
      ret = builder.create<AffineVectorLoadOp>(
          loadOp.getLoc(), ArrayRef<Type>{vecType}, operand, loadOp.indices());
      auto mapAttr = AffineMapAttr::get(loadOp.getAffineMap());
      // Set the map attribute
      ret->setAttr(AffineVectorLoadOp::getMapAttrName(), mapAttr);
      IVLOG(1, "Lubo: New LOAD op:" << mlir::debugString(vecType) << ":"
                                    << mlir::debugString(*ret));
    } else {
      llvm_unreachable("Vectorize: Invalid stride for AffineLoadOp");
    }
    return ret;
  }

  Operation *vectorizeReduceOp(Operation *op, OpBuilder &builder) {
    IVLOG(1, "Lubo: New REDUCE op111:" << mlir::debugString(vecType) << ":"
                                       << mlir::debugString(*op));
    AffineReduceOp reduceOp = dyn_cast_or_null<AffineReduceOp>(op);
    if (!reduceOp) {
      llvm_unreachable("Vectorize: Invalid state for AffineReduceOp");
    }

    Value mem = reduceOp.mem();    // Lubo mappedValue(reduceOp.mem());
    Value vector = reduceOp.val(); // Lubo mappedValue(reduceOp.val());
    Type tp = vector.getType();    // Lubo
    IVLOG(1, "Lubo: New REDUCE op222:" << mlir::debugString(tp));
    Type tp1 = mem.getType(); // Lubo
    IVLOG(1, "Lubo: New REDUCE op333:" << mlir::debugString(tp1));
    Operation *ret = builder.create<AffineVectorReduceOp>(
        reduceOp.getLoc(), ArrayRef<Type>{mem.getType()}, reduceOp.agg(),
        vector, mem, reduceOp.map(), reduceOp.idxs());
    Type tttp = ret->getResults()[0].getType();
    IVLOG(1, "Lubo: New REDUCE op:" << mlir::debugString(vecType) << ":"
                                    << mlir::debugString(*ret) << ":"
                                    << ret->getResults().size() << ":"
                                    << mlir::debugString(tttp));
    return ret;
  }

  Operation *vectorizeOperation(OpBuilder &builder, Operation &loopOperation) {
    auto pair = vectorizableOps.find(&loopOperation);
    IVLOG(1, "Lubo: Processing loop operation:"
                 << loopOperation.getResultTypes().size() << ":"
                 << mlir::debugString(loopOperation));
    if (pair != vectorizableOps.end()) {
      OpVectState opVectState = pair->second;
      IVLOG(1, "Lubo: Found operation:" << opVectState.opType << ":"
                                        << opVectState.stride);

      // TODO: transform

      if (opVectState.opType == OpType::SCALAR) {
        Operation *ret = vectorizeScalarOp(&loopOperation, builder);
        mapResults(loopOperation, ret);
        Type tp = ret->getResultTypes()[0]; // Lubo
        IVLOG(1, "Lubo: Transformed operation SCALAR:"
                     << ret->getResultTypes().size() << ":"
                     << mlir::debugString(tp) << ":"
                     << mlir::debugString(*ret));
        return ret;
      } else if (opVectState.opType == OpType::LOAD) {
        Operation *ret = vectorizeLoadOp(&loopOperation, opVectState, builder);
        mapResults(loopOperation, ret);
        IVLOG(1, "Lubo: Transformed operation LOAD:"
                     << ret->getResultTypes().size() << ":"
                     << mlir::debugString(*ret));
        return ret;
      } else if (opVectState.opType == OpType::REDUCE) {
        Operation *ret = vectorizeReduceOp(&loopOperation, builder);
        mapResults(loopOperation, ret);
        IVLOG(1, "Lubo: Transformed operation REDUCE:"
                     << ret->getResultTypes().size() << ":"
                     << mlir::debugString(*ret));
        return ret;
      } else {
        llvm_unreachable("Vectorize: Unexpected vectorizable operation type");
      }
    }

    Operation *ret = builder.clone(loopOperation);
    mapResults(loopOperation, ret);
    IVLOG(1, "Lubo: Cloned operation:" << ret->getResultTypes().size() << ":"
                                       << mlir::debugString(*ret));
    return ret;
  }

  void createAndPopulateNewLoop(AffineParallelOp op, unsigned argNum) {
    // Lubo mlir::Block *block = op.getOperation()->getBlock();
    // mlir::OpBuilder builder(op); // Lubo , op.getBody()->begin());
    IVLOG(1, "Lubo10: " << mlir::debugString(*op.getParentOp()->getParentOp()));

    // Lubo Operation* luboOp = op.getParentOp()->getParentOp();
    mlir::OpBuilder builder(op);
    // Fix up the steps
    std::vector<Attribute> newSteps;
    for (unsigned int i = 0; i < op.getIVs().size(); i++) {
      int64_t val = op.steps().getValue()[i].cast<IntegerAttr>().getInt() *
                    (i == argNum ? numElementsInRegister : 1);
      newSteps.push_back(builder.getI64IntegerAttr(val));
    }
    SmallVector<AtomicRMWKind, 8> reductions(op.getResultTypes().size(),
                                             AtomicRMWKind::assign);

    SmallVector<Attribute, 8> reductionAttrs;
    for (AtomicRMWKind reduction : reductions)
      reductionAttrs.push_back(
          builder.getI64IntegerAttr(static_cast<int64_t>(reduction)));

    AffineParallelOp newAffineParallelOp = builder.create<AffineParallelOp>(
        op.getLoc(), op.getResultTypes(),
        ArrayAttr::get(reductionAttrs, op.getContext()),
        op.getLowerBoundsValueMap()
            .getAffineMap(), // Lubo op.getLowerBoundsOperands(),
        op.getUpperBoundsValueMap()
            .getAffineMap(), // Lubo op.getUpperBoundsOperands(),
        builder.getArrayAttr(newSteps), op.mapOperands());
    IVLOG(1,
          "Lubo8: " << mlir::debugString(*newAffineParallelOp.getOperation()));
    // Lubo for (auto it = block->begin(); it != block->end(); it++) {
    IVLOG(1, "Lubo4: " << mlir::debugString(*op.getOperation()));
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
    // Lubo }
    // Lubo}
    op.replaceAllUsesWith(newAffineParallelOp);
    // Lubo
    mlir::Block *block = op.getOperation()->getBlock();
    for (auto it = block->begin(); it != block->end(); it++) {
      IVLOG(1, "Lubo40: " << mlir::debugString(*it));
    }

    // Lubo end

    IVLOG(1, "Lubo10.2: ");
    // IVLOG(1, "Lubo10.1: " <<
    // mlir::debugString(*op.getParentOp()->getParentOp()));
    IVLOG(1, "Lubo10.3: ");
    // Remove the original ForOp
    mlir::Region *parentRegion = op.getOperation()->getParentRegion();
    if (!parentRegion) {
      llvm_unreachable("AffineParallelOp with no parent region.");
    }

    // Lubo IVLOG(1, "Lubo14: " << mlir::debugString(*luboOp));
    auto ivss = op.getIVs(); // Lubo
    for (unsigned int i = 0; i < ivss.size(); i++) {
      ivss[i].getOwner()->dropAllReferences();
      ivss[i].getOwner()->dropAllDefinedValueUses();
      // ivss[i].getOwner()->erase();
      IVLOG(1, "Lubo10.1: " << ivss[i].getOwner());
    }

    op.erase();
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

    for (auto it = block->begin(); it != block->end(); it++) {
      IVLOG(1, "Lubo41: " << mlir::debugString(*it));
    }
    IVLOG(1, "Lubo11: "); // Lubo  << mlir::debugString(*luboOp));
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

    IVLOG(1, "Lubo vecType: " << mlir::debugString(vecType));

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
    return true;
  }
};

bool performVectorization(AffineParallelOp op, BlockArgument index,
                          unsigned vectorSize, unsigned minElemWidth) {
  Impl impl(op, index, vectorSize, minElemWidth);
  return impl.vectorize();
}
} // namespace pmlc::dialect::pxa
