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
private:
  AffineParallelOp loop;
  BlockArgument index;
  unsigned vectorSize;
  DenseSet<Value> vectorizedValues;
  DenseSet<Operation *> vectorizedOps;

  bool tryVectorizeOperation(Operation *op) {
    return llvm::TypeSwitch<Operation *, bool>(op)
        .Case<AffineLoadOp>([&](auto op) {
          auto strideInfo = computeStrideInfo(op);
          if (!strideInfo) {
            IVLOG(1, "Vectorize: Failed, non-affine strides");
            return false;
          }

          auto pair = strideInfo->strides.find(index);
          if (pair == strideInfo->strides.end()) {
            IVLOG(1, "Vectorize: AffineLoadOp stride of 0 for index");
            return true;
          }

          if (pair->second != 1) {
            IVLOG(1, "Vectorize: Failed, AffineLoadOp stride different "
                     "than 0 and 1 for index:"
                         << pair->second);
            return false;
          }
          vectorizedOps.insert(op);
          vectorizedValues.insert(op.getResult());
          return true;
        })
        .Case<AffineReduceOp>([&](auto op) {
          auto strideInfo = computeStrideInfo(op);
          if (!strideInfo) {
            IVLOG(1, "Vectorize: Failed, non-affine strides");
            return false;
          }

          // TODO: Deal with reductions of stride 0.
          // TODO: Call vector_reduce and then scalar pxa.reduce.
          auto pair = strideInfo->strides.find(index);
          if (pair == strideInfo->strides.end() || pair->second != 1) {
            IVLOG(1, "Vectorize: Failed, AffineReduceOp stride != 1 for index");
            return false;
          }

          vectorizedOps.insert(op);
          return true;
        })
        .Default([&](Operation *op) {
          if (op->getNumRegions() != 0) {
            IVLOG(1, "Vectorize: Failed, interior loops");
            return false;
          }
          if (!mlir::isa<VectorUnrollOpInterface>(op)) {
            // Probably not a vectorizable op.  Verify it doesn't use an
            // vectorized results.
            for (auto operand : op->getOperands()) {
              if (vectorizedValues.count(operand)) {
                IVLOG(1,
                      "Vectorize: Failed, unknown op used vectorized result");
                return false;
              }
            }
            // Otherwise, safe and ignorable.
            return true;
          }
          // Only vectorize if at least one operand is vectorized
          bool anyVec = false;
          for (auto operand : op->getOperands()) {
            if (vectorizedValues.count(operand)) {
              anyVec = true;
            }
          }
          if (!anyVec) {
            // No need to vectorize, all is good
            return true;
          }
          // We also don't handle ops with multiple results
          if (op->getNumResults() != 1) {
            IVLOG(1, "Vectorize: Failed, multi-result scalar op");
            return false;
          }
          vectorizedOps.insert(op);
          vectorizedValues.insert(op->getResult(0));
          return true;
        });
  }

public:
  Impl(AffineParallelOp loop, BlockArgument index, unsigned vectorSize)
      : loop(loop), index(index), vectorSize(vectorSize) {}

  void vectorizeScalarOp(Operation *op) {
    OpBuilder builder(op);
    for (auto &operand : op->getOpOperands()) {
      // For each non-vector operand, broadcast as needed
      if (!operand.get().getType().isa<VectorType>()) {
        auto vecType = VectorType::get({vectorSize}, operand.get().getType());
        auto bcast = builder.create<vector::BroadcastOp>(op->getLoc(), vecType,
                                                         operand.get());
        operand.set(bcast);
      }
    }
    // Update the result type
    auto result = op->getResult(0);
    auto vecType = VectorType::get({vectorSize}, result.getType());
    result.setType(vecType);
  }

  void vectorizeLoadOp(AffineLoadOp op) {
    Value operand = op.getMemRef();
    auto eltType = op.getMemRefType().getElementType();
    auto vecType = VectorType::get(vectorSize, eltType);
    OpBuilder builder(op);
    auto vecOp = builder.create<AffineVectorLoadOp>(
        op.getLoc(), ArrayRef<Type>{vecType}, operand, op.indices());
    // TODO: Add support for direct construction with map to VectorLoadOp
    auto mapAttr = AffineMapAttr::get(op.getAffineMap());
    vecOp.setAttr(AffineVectorLoadOp::getMapAttrName(), mapAttr);
    op.replaceAllUsesWith(vecOp.getResult());
    op.erase();
  }

  void vectorizeReduceOp(AffineReduceOp op) {
    Value mem = op.mem();
    Value vector = op.val();
    Type vecType = vector.getType();
    assert(vecType.isa<VectorType>());
    OpBuilder builder(op);
    auto vecOp = builder.create<AffineVectorReduceOp>(
        op.getLoc(), ArrayRef<Type>{mem.getType()}, op.agg(), vector, mem,
        op.map(), op.idxs());
    op.replaceAllUsesWith(vecOp.getResult());
    op.erase();
  }

  void vectorizeOperation(Operation *op) {
    if (!vectorizedOps.count(op)) {
      return;
    }
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      vectorizeLoadOp(loadOp);
    } else if (auto reduceOp = dyn_cast<AffineReduceOp>(op)) {
      vectorizeReduceOp(reduceOp);
    } else {
      vectorizeScalarOp(op);
    }
  }

  bool vectorize() {
    mlir::Block *body = loop.getBody();
    IVLOG(1, "Vectorize: Attempting Vectorizing for BlockArgument: "
                 << index.getArgNumber());

    auto ranges = loop.getConstantRanges();
    if (!ranges) {
      IVLOG(1, "Cannot Vectorize: Requires constant ranges");
      return false;
    }

    auto argNum = index.getArgNumber();
    if ((*ranges)[argNum] % vectorSize != 0) {
      IVLOG(1, "Cannot Vectorize: The dimension being vectorized not multiple "
               "of the number of elements in a register");
      return false;
    }

    auto step = loop.steps().getValue()[argNum].cast<IntegerAttr>().getInt();
    if (step != 1) {
      IVLOG(1, "Cannot Vectorize: The steps for the dimension being vectorized "
               "is not 1");
      return false;
    }

    bool vectorizable = true;
    body->walk(
        [&](Operation *op) { vectorizable &= tryVectorizeOperation(op); });
    if (!vectorizable) {
      IVLOG(1, "Found an unvectorizable op");
      return false;
    }

    // Preflight complete, do the transform
    for (auto &op : llvm::make_early_inc_range(body->getOperations())) {
      vectorizeOperation(&op);
    }
    llvm::SmallVector<int64_t, 6> steps;
    for (auto ia : loop.steps().cast<ArrayAttr>().getValue()) {
      steps.push_back(ia.cast<IntegerAttr>().getInt());
    }
    steps[argNum] *= vectorSize;
    loop.setSteps(steps);
    return true;
  }
};

bool performVectorization(AffineParallelOp op, BlockArgument index,
                          unsigned vectorSize) {
  Impl impl(op, index, vectorSize);
  return impl.vectorize();
}
} // namespace pmlc::dialect::pxa
