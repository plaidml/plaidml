// Copyright 2020 Intel Corporation

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Interfaces/VectorInterfaces.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
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

  LogicalResult tryVectorizeOperation(Operation *op) {
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<AffineLoadOp>([&](auto op) {
          auto strideInfo = computeStrideInfo(op);
          if (!strideInfo) {
            IVLOG(3, "Vectorize: Failed, non-affine strides");
            return failure();
          }

          auto it = strideInfo->strides.find(index);
          if (it == strideInfo->strides.end()) {
            // Stride 0, safe to leave unvectorized
            return success();
          }

          // Stride is non-zero, must vectorize
          if (it->second != 1) {
            IVLOG(3, "Vectorize: Failed, AffineLoadOp stride != 1");
            return failure();
          }
          vectorizedOps.insert(op);
          vectorizedValues.insert(op.getResult());
          return success();
        })
        .Case<AffineReduceOp>([&](auto op) {
          auto strideInfo = computeStrideInfo(op);
          if (!strideInfo) {
            IVLOG(3, "Vectorize: Failed, non-affine strides");
            return failure();
          }

          auto it = strideInfo->strides.find(index);
          if (it == strideInfo->strides.end()) {
            // Deal with reductions of stride 0.
            // TODO: If input is a vector, call vector.reduce and then scalar
            // pxa.reduce. Right now, we say things are cool if out input isn't
            // vectorized
            return failure(vectorizedValues.count(op.val()));
          }
          if (it->second != 1) {
            IVLOG(3, "Vectorize: Failed, AffineReduceOp stride != 1");
            return failure();
          }

          vectorizedOps.insert(op);
          return success();
        })
        .Default([&](Operation *op) {
          if (op->getNumRegions() != 0) {
            IVLOG(3, "Vectorize: Failed, interior loops");
            return failure();
          }
          if (!mlir::isa<VectorUnrollOpInterface>(op)) {
            // Probably not a vectorizable op.  Verify it doesn't use an
            // vectorized results.
            for (auto operand : op->getOperands()) {
              if (vectorizedValues.count(operand)) {
                IVLOG(3,
                      "Vectorize: Failed, unknown op used vectorized result");
                return failure();
              }
            }
            // Otherwise, safe and ignorable.
            return success();
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
            return success();
          }
          // We also don't handle ops with multiple results
          if (op->getNumResults() != 1) {
            IVLOG(3, "Vectorize: Failed, multi-result scalar op");
            return failure();
          }
          vectorizedOps.insert(op);
          vectorizedValues.insert(op->getResult(0));
          return success();
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
    Value val = op.val();
    OpBuilder builder(op);
    if (!val.getType().isa<VectorType>()) {
      auto vecType = VectorType::get({vectorSize}, val.getType());
      auto bcast =
          builder.create<vector::BroadcastOp>(op.getLoc(), vecType, val);
      val = bcast.getResult();
    }
    auto vecOp = builder.create<AffineVectorReduceOp>(
        op.getLoc(), ArrayRef<Type>{mem.getType()}, op.agg(), val, mem,
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

  LogicalResult vectorize() {
    mlir::Block *body = loop.getBody();
    IVLOG(3, "Vectorize: Attempting Vectorizing for BlockArgument: "
                 << index.getArgNumber());

    auto ranges = loop.getConstantRanges();
    if (!ranges) {
      IVLOG(3, " Vectorize: Failed, Requires constant ranges");
      return failure();
    }

    auto argNum = index.getArgNumber();
    if ((*ranges)[argNum] % vectorSize != 0) {
      IVLOG(3, "Vectorize: Failed, the dimension being vectorized not multiple "
               "of the number of elements in a register");
      return failure();
    }

    auto step = loop.steps().getValue()[argNum].cast<IntegerAttr>().getInt();
    if (step != 1) {
      IVLOG(3,
            "Vectorize: Failed, the steps for the dimension being vectorized "
            "is not 1");
      return failure();
    }

    bool vectorizable = true;
    for (auto &op : body->getOperations()) {
      vectorizable &= succeeded(tryVectorizeOperation(&op));
    }
    if (!vectorizable) {
      return failure();
    }
    if (vectorizedOps.empty()) {
      // TODO: should we actually fail in this case?  Currently we need to since
      // we have no cost model
      IVLOG(3, "Vectorize: Failed, no point in vectorization");
      return failure();
    }

    // Preflight complete, do the transform
    for (auto &op : llvm::make_early_inc_range(body->getOperations())) {
      vectorizeOperation(&op);
    }
    // TODO: We should upstream a utility like getSteps since this code is
    // duplicated in multiple places
    llvm::SmallVector<int64_t, 6> steps;
    for (auto stepAttr : loop.steps().cast<ArrayAttr>().getValue()) {
      steps.push_back(stepAttr.cast<IntegerAttr>().getInt());
    }
    steps[argNum] *= vectorSize;
    loop.setSteps(steps);
    return success();
  }
};

LogicalResult performVectorization(AffineParallelOp op, BlockArgument index,
                                   unsigned vectorSize) {
  Impl impl(op, index, vectorSize);
  return impl.vectorize();
}

struct VectorizeExample : public VectorizeExampleBase<VectorizeExample> {
  void runOnFunction() final {
    static constexpr unsigned vectorWidth = 8;
    auto func = getFunction();
    // Vectorize only the outermost loops
    for (auto &op : func.getBody().front()) {
      auto loop = mlir::dyn_cast<mlir::AffineParallelOp>(op);
      if (!loop) {
        continue;
      }
      // Try IV's until we succeed
      for (unsigned int i = 0; i < loop.getIVs().size(); i++) {
        auto blockArg = loop.getIVs()[i];
        if (succeeded(performVectorization(loop, blockArg, vectorWidth))) {
          break;
        }
      }
    }
  }
};

std::unique_ptr<mlir::Pass> createVectorizeExamplePass() {
  return std::make_unique<VectorizeExample>();
}

} // namespace pmlc::dialect::pxa
