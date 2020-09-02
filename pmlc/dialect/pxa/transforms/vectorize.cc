// Copyright 2020 Intel Corporation

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Interfaces/VectorInterfaces.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/vectorize.h"
#include "pmlc/util/logging.h"

#include "mlir/Support/DebugStringHelper.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::pxa {
using pmlc::dialect::pxa::PxaReduceOp;

class Impl {
private:
  AffineParallelOp loop;
  BlockArgument index;
  unsigned vectorSize;
  DenseSet<Value> vectorizedValues;
  DenseSet<Operation *> vectorizedOps;
  llvm::DenseSet<Operation *> zeroStrideReductions;

  const char *stringifyAtomicRMWKindForVectorReductionOp(AtomicRMWKind val) {
    switch (val) {
    case AtomicRMWKind::addf:
      return "add";
    case AtomicRMWKind::addi:
      return "add";
    case AtomicRMWKind::assign:
      return "invalid";
    case AtomicRMWKind::maxf:
      return "max";
    case AtomicRMWKind::maxs:
      return "max";
    case AtomicRMWKind::maxu:
      return "max";
    case AtomicRMWKind::minf:
      return "min";
    case AtomicRMWKind::mins:
      return "min";
    case AtomicRMWKind::minu:
      return "min";
    case AtomicRMWKind::mulf:
      return "mul";
    case AtomicRMWKind::muli:
      return "mul";
    }
    llvm_unreachable("Invalid aggregation type");
  }

  LogicalResult tryVectorizeOperation(Operation *op) {
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<PxaLoadOp>([&](auto op) {
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
        .Case<PxaReduceOp>([&](auto op) {
          auto strideInfo = computeStrideInfo(op);
          if (!strideInfo) {
            IVLOG(3, "Vectorize: Failed, non-affine strides");
            return failure();
          }

          auto it = strideInfo->strides.find(index);
          if (it == strideInfo->strides.end()) {
            // vector::ReductionOp doesn't support pxa's assign reduction.
            // Also, make sure we handle only the supported types -
            // see the vector::ReductionOp verification code
            // (https://github.com/llvm/llvm-project/blob/master/mlir/lib/Dialect/Vector/VectorOps.cpp#L134).
            Type eltType = op.getMemRefType().getElementType();
            if (op.agg() == AtomicRMWKind::assign ||
                (!eltType.isF32() && !eltType.isF64() &&
                 !eltType.isSignlessInteger(32) &&
                 !eltType.isSignlessInteger(64))) {
              op.emitRemark("Vectorization failed: Unsupported reduction or "
                            "type for vector::ReductionOp");
              return failure();
            }
            // If stride is 0, "remember it" as such.
            zeroStrideReductions.insert(op.getOperation());
          } else if (it->second != 1) {
            IVLOG(3, "Vectorize: Failed, PxaReduceOp stride != 1");
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

  void vectorizeLoadOp(PxaLoadOp op) {
    Value operand = op.getMemRef();
    auto eltType = op.getMemRefType().getElementType();
    auto vecType = VectorType::get(vectorSize, eltType);
    OpBuilder builder(op);
    auto vecOp = builder.create<PxaVectorLoadOp>(
        op.getLoc(), vecType, operand, op.getAffineMap(), op.getMapOperands());
    op.replaceAllUsesWith(vecOp.getResult());
    op.erase();
  }

  void vectorizeReduceOp(PxaReduceOp op) {
    Value val = op.val();
    OpBuilder builder(op);
    if (!val.getType().isa<VectorType>()) {
      auto vecType = VectorType::get({vectorSize}, val.getType());
      auto bcast =
          builder.create<vector::BroadcastOp>(op.getLoc(), vecType, val);
      val = bcast.getResult();
    }
    // Add vector_reduction only if the stride is 0
    if (zeroStrideReductions.find(op.getOperation()) !=
        zeroStrideReductions.end()) {
      vector::ReductionOp reductionOp = builder.create<vector::ReductionOp>(
          op.getLoc(), op.getMemRefType().getElementType(),
          builder.getStringAttr(
              stringifyAtomicRMWKindForVectorReductionOp(op.agg())),
          val, ValueRange{});

      auto reduceOp = builder.create<PxaReduceOp>(
          op.getLoc(), ArrayRef<Type>{op.getMemRefType()}, op.agg(),
          reductionOp.getResult(), op.memref(), op.map(), op.idxs());
      op.replaceAllUsesWith(reduceOp.getResult());
      op.erase();
    } else {
      auto vecOp = builder.create<PxaVectorReduceOp>(
          op.getLoc(), ArrayRef<Type>{op.getMemRefType()}, op.agg(), val,
          op.memref(), op.map(), op.idxs());
      op.replaceAllUsesWith(vecOp.getResult());
      op.erase();
    }
  }

  void vectorizeOperation(Operation *op) {
    if (!vectorizedOps.count(op)) {
      return;
    }
    if (auto loadOp = dyn_cast<PxaLoadOp>(op)) {
      vectorizeLoadOp(loadOp);
    } else if (auto reduceOp = dyn_cast<PxaReduceOp>(op)) {
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

    auto steps = loop.getSteps();
    auto step = steps[argNum];
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

LogicalResult simpleVectorize(AffineParallelOp op, unsigned vecSize) {
  if (op.getNumResults() != 1) {
    return failure();
  }
  auto reduce = mlir::dyn_cast<PxaReduceOp>(getPrevWriter(op.getResult(0)));
  if (!reduce) {
    return failure();
  }
  auto maybeSI = computeStrideInfo(reduce);
  if (!maybeSI) {
    return failure();
  }
  SmallVector<BlockArgument, 4> options;
  for (auto ba : op.getIVs()) {
    if (maybeSI->strides.count(ba) && maybeSI->strides[ba] == 1) {
      options.push_back(ba);
    }
  }
  if (options.size() != 1) {
    return failure();
  }
  return performVectorization(op, options[0], vecSize);
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

// TODO: Maybe move this to a generic utility somewhere
template <typename OpTy, typename... Args>
static OpTy replaceOp(Operation *op, Args &&... args) {
  OpBuilder builder(op);
  auto newOp = builder.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
  op->getResult(0).replaceAllUsesWith(newOp.getResult());
  op->erase();
  return newOp;
}

LogicalResult vectorizeBuffer(AllocOp op) {
  // Verify that all uses are vector load/stores of the same width and with
  // valid minimum strides
  int64_t vecSize = 0;
  // Make generic lambda to verify/capture vector size
  auto validAccess = [&](auto vecOp) -> LogicalResult {
    auto shape = vecOp.getVectorType().getShape();
    if (shape.size() != 1) {
      return failure(); // Only support 1-d vectors
    }
    int64_t newSize = shape[0];
    if (vecSize && vecSize != newSize) {
      return failure(); // All vectors must have the same width
    }
    vecSize = newSize;
    assert(vecSize != 0 && "Vector shape should never be zero elements");
    auto maybeStride = computeStrideInfo(vecOp);
    if (!maybeStride) {
      return failure(); // Non strided access
    }
    auto range = maybeStride->range();
    if (range.stride % vecSize != 0) {
      return failure(); // Vector op not aligned
    }
    return success();
  };
  // Call the lambda on all uses + verify all are are valid vector ops
  for (auto &use : getIndirectAccessUses(op)) {
    if (auto vecOp = dyn_cast<PxaVectorLoadOp>(use.getOwner())) {
      if (failed(validAccess(vecOp))) {
        return failure();
      }
    } else if (auto vecOp = dyn_cast<PxaVectorReduceOp>(use.getOwner())) {
      if (failed(validAccess(vecOp))) {
        return failure();
      }
    } else {
      return failure(); // Non vector access detected
    }
  }
  // Exit early if no accesses
  if (!vecSize) {
    return failure();
  }
  // Compute new memref shape
  auto mtype = op.getType();
  auto mshape = mtype.getShape();
  if (mshape.size() < 1) {
    return failure(); // Can't perform transform on a scalar
  }
  SmallVector<int64_t, 4> newShape;
  for (size_t i = 0; i < mshape.size(); i++) {
    if (i == mshape.size() - 1) {
      // Last index, verify it's divisible vector size + reduce
      if (mshape[i] % vecSize != 0) {
        return failure();
      }
      newShape.push_back(mshape[i] / vecSize);
    } else {
      // Leave non-final indexes alone
      newShape.push_back(mshape[i]);
    }
  }
  // Make the new type
  auto newType = MemRefType::get(
      newShape, VectorType::get({vecSize}, mtype.getElementType()));
  // Replace the alloc
  auto newOp = replaceOp<AllocOp>(op, newType);
  // Walk over the uses and update them all
  auto curUse = IndirectUsesIterator(newOp);
  while (curUse != IndirectUsesIterator()) {
    curUse->get().setType(newType);
    if (auto vecOp = dyn_cast<PxaVectorLoadOp>(curUse->getOwner())) {
      replaceOp<PxaLoadOp>(vecOp, vecOp.getMemRef(), vecOp.getAffineMap(),
                           vecOp.getMapOperands());
      curUse++;
    } else if (auto vecOp = dyn_cast<PxaVectorReduceOp>(curUse->getOwner())) {
      auto newVecOp = replaceOp<PxaReduceOp>(
          vecOp, vecOp.agg(), vecOp.getValueToStore(), vecOp.getMemRef(),
          vecOp.getAffineMap(), vecOp.getMapOperands());
      curUse = IndirectUsesIterator(newVecOp);
    } else {
      curUse++;
    }
  }
  return success();
}

} // namespace pmlc::dialect::pxa
