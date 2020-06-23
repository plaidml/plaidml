//===- SCFToGPU.cpp - Convert an affine loop nest to a GPU kernel -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a straightforward conversion of an loop nest into a GPU
// kernel.  The caller is expected to guarantee that the conversion is correct
// or to further transform the kernel to ensure correctness.
//
//===----------------------------------------------------------------------===//

#include "pmlc/conversion/SCFToGPU/SCFToGPU.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "loops-to-gpu"

using namespace mlir;      // NOLINT
using namespace mlir::scf; // NOLINT

using llvm::seq;

// Extract an indexed value from KernelDim3.
static Value getDim3Value(const gpu::KernelDim3 &dim3, unsigned pos) {
  switch (pos) {
  case 0:
    return dim3.x;
  case 1:
    return dim3.y;
  case 2:
    return dim3.z;
  default:
    llvm_unreachable("dim3 position out of bounds");
  }
  return nullptr;
}

// Get the lower bound-related operands of a loop operation.
static Operation::operand_range getLowerBoundOperands(AffineForOp forOp) {
  return forOp.getLowerBoundOperands();
}
static SmallVector<Value, 1> getLowerBoundOperands(ForOp forOp) {
  SmallVector<Value, 1> bounds(1, forOp.lowerBound());
  return bounds;
}

// Get the upper bound-related operands of a loop operation.
static Operation::operand_range getUpperBoundOperands(AffineForOp forOp) {
  return forOp.getUpperBoundOperands();
}
static SmallVector<Value, 1> getUpperBoundOperands(ForOp forOp) {
  SmallVector<Value, 1> bounds(1, forOp.upperBound());
  return bounds;
}

// Get a Value that corresponds to the loop step.  If the step is an attribute,
// materialize a corresponding constant using builder.
static Value getOrCreateStep(AffineForOp forOp, OpBuilder &builder) {
  return builder.create<ConstantIndexOp>(forOp.getLoc(), forOp.getStep());
}
static Value getOrCreateStep(ForOp forOp, OpBuilder &) { return forOp.step(); }

// Get a Value for the loop lower bound.  If the value requires computation,
// materialize the instructions using builder.
static Value getOrEmitLowerBound(AffineForOp forOp, OpBuilder &builder) {
  return lowerAffineLowerBound(forOp, builder);
}
static Value getOrEmitLowerBound(ForOp forOp, OpBuilder &) {
  return forOp.lowerBound();
}

// Get a Value for the loop upper bound.  If the value requires computation,
// materialize the instructions using builder.
static Value getOrEmitUpperBound(AffineForOp forOp, OpBuilder &builder) {
  return lowerAffineUpperBound(forOp, builder);
}
static Value getOrEmitUpperBound(ForOp forOp, OpBuilder &) {
  return forOp.upperBound();
}

// Check the structure of the loop nest:
//   - there are enough loops to map to numDims;
//   - the loops are perfectly nested;
//   - the loop bounds can be computed above the outermost loop.
// This roughly corresponds to the "matcher" part of the pattern-based
// rewriting infrastructure.
template <typename OpTy>
static LogicalResult checkLoopNestMappableImpl(OpTy forOp, unsigned numDims) {
  Region &limit = forOp.region();
  for (unsigned i = 0, e = numDims; i < e; ++i) {
    Operation *nested = &forOp.getBody()->front();
    if (!areValuesDefinedAbove(getLowerBoundOperands(forOp), limit) ||
        !areValuesDefinedAbove(getUpperBoundOperands(forOp), limit))
      return forOp.emitError(
          "loops with bounds depending on other mapped loops "
          "are not supported");

    // The innermost loop can have an arbitrary body, skip the perfect nesting
    // check for it.
    if (i == e - 1)
      break;

    auto begin = forOp.getBody()->begin(), end = forOp.getBody()->end();
    if (forOp.getBody()->empty() || std::next(begin, 2) != end)
      return forOp.emitError("expected perfectly nested loops in the body");

    if (!(forOp = dyn_cast<OpTy>(nested)))
      return nested->emitError("expected a nested loop");
  }
  return success();
}

template <typename OpTy>
static LogicalResult checkLoopNestMappable(OpTy forOp, unsigned numBlockDims,
                                           unsigned numThreadDims) {
  if (numBlockDims < 1 || numThreadDims < 1) {
    LLVM_DEBUG(llvm::dbgs() << "nothing to map");
    return success();
  }

  if (numBlockDims > 3) {
    return forOp.emitError("cannot map to more than 3 block dimensions");
  }
  if (numThreadDims > 3) {
    return forOp.emitError("cannot map to more than 3 thread dimensions");
  }
  return checkLoopNestMappableImpl(forOp, numBlockDims + numThreadDims);
}

template <typename OpTy>
static LogicalResult checkLoopOpMappable(OpTy forOp, unsigned numBlockDims,
                                         unsigned numThreadDims) {
  if (numBlockDims < 1 || numThreadDims < 1) {
    LLVM_DEBUG(llvm::dbgs() << "nothing to map");
    return success();
  }

  if (numBlockDims > 3) {
    return forOp.emitError("cannot map to more than 3 block dimensions");
  }
  if (numThreadDims > 3) {
    return forOp.emitError("cannot map to more than 3 thread dimensions");
  }
  if (numBlockDims != numThreadDims) {
    // TODO(ravishankarm) : This can probably be relaxed by having a one-trip
    // loop for the missing dimension, but there is not reason to handle this
    // case for now.
    return forOp.emitError(
        "mismatch in block dimensions and thread dimensions");
  }

  // Check that the forOp contains perfectly nested loops for numBlockDims
  if (failed(checkLoopNestMappableImpl(forOp, numBlockDims))) {
    return failure();
  }

  // Get to the innermost loop.
  for (auto i : seq<unsigned>(0, numBlockDims - 1)) {
    forOp = cast<OpTy>(&forOp.getBody()->front());
    (void)i;
  }

  // The forOp now points to the body of the innermost loop mapped to blocks.
  for (Operation &op : *forOp.getBody()) {
    // If the operation is a loop, check that it is mappable to workItems.
    if (auto innerLoop = dyn_cast<OpTy>(&op)) {
      if (failed(checkLoopNestMappableImpl(innerLoop, numThreadDims))) {
        return failure();
      }
      continue;
    }
    // TODO(ravishankarm) : If it is not a loop op, it is assumed that the
    // statement is executed by all threads. It might be a collective operation,
    // or some non-side effect instruction. Have to decide on "allowable"
    // statements and check for those here.
  }
  return success();
}

namespace {
// Helper structure that holds common state of the loop to GPU kernel
// conversion.
struct LoopToGpuConverter {
  template <typename OpTy>
  Optional<OpTy> collectBounds(OpTy forOp, unsigned numLoops);

  template <typename OpTy>
  void createLaunch(OpTy rootForOp, OpTy innermostForOp, unsigned numBlockDims,
                    unsigned numThreadDims);

  // Ranges of the loops mapped to blocks or threads.
  SmallVector<Value, 6> dims;
  // Lower bounds of the loops mapped to blocks or threads.
  SmallVector<Value, 6> lbs;
  // Induction variables of the loops mapped to blocks or threads.
  SmallVector<Value, 6> ivs;
  // Steps of the loops mapped to blocks or threads.
  SmallVector<Value, 6> steps;
};
} // namespace

// Return true if the value is obviously a constant "one".
static bool isConstantOne(Value value) {
  if (auto def = value.getDefiningOp<ConstantIndexOp>())
    return def.getValue() == 1;
  return false;
}

// Collect ranges, bounds, steps and induction variables in preparation for
// mapping a loop nest of depth "numLoops" rooted at "forOp" to a GPU kernel.
// This may fail if the IR for computing loop bounds cannot be constructed, for
// example if an affine loop uses semi-affine maps. Return the last loop to be
// mapped on success, llvm::None on failure.
template <typename OpTy>
Optional<OpTy> LoopToGpuConverter::collectBounds(OpTy forOp,
                                                 unsigned numLoops) {
  OpBuilder builder(forOp.getOperation());
  dims.reserve(numLoops);
  lbs.reserve(numLoops);
  ivs.reserve(numLoops);
  steps.reserve(numLoops);
  OpTy currentLoop = forOp;
  for (unsigned i = 0; i < numLoops; ++i) {
    Value lowerBound = getOrEmitLowerBound(currentLoop, builder);
    Value upperBound = getOrEmitUpperBound(currentLoop, builder);
    if (!lowerBound || !upperBound) {
      return llvm::None;
    }

    Value range =
        builder.create<SubIOp>(currentLoop.getLoc(), upperBound, lowerBound);
    Value step = getOrCreateStep(currentLoop, builder);
    if (!isConstantOne(step))
      range = builder.create<SignedDivIOp>(currentLoop.getLoc(), range, step);
    dims.push_back(range);

    lbs.push_back(lowerBound);
    ivs.push_back(currentLoop.getInductionVar());
    steps.push_back(step);

    if (i != numLoops - 1)
      currentLoop = cast<OpTy>(&currentLoop.getBody()->front());
  }
  return currentLoop;
}

/// Given `nDims` perfectly nested loops rooted as `rootForOp`, convert them o
/// be partitioned across workgroups or workitems. The values for the
/// workgroup/workitem id along each dimension is passed in with `ids`. The
/// number of workgroups/workitems along each dimension are passed in with
/// `nids`. The innermost loop is mapped to the x-dimension, followed by the
/// next innermost loop to y-dimension, followed by z-dimension.
template <typename OpTy>
static OpTy createGPULaunchLoops(OpTy rootForOp, ArrayRef<Value> ids,
                                 ArrayRef<Value> nids) {
  auto nDims = ids.size();
  assert(nDims == nids.size());
  for (auto dim : llvm::seq<unsigned>(0, nDims)) {
    // TODO(ravishankarm): Don't always need to generate a loop here. If nids >=
    // number of iterations of the original loop, this becomes a if
    // condition. Though that does rely on how the workgroup/workitem sizes are
    // specified to begin with.
    mapLoopToProcessorIds(rootForOp, ids[dim], nids[dim]);
    if (dim != nDims - 1) {
      rootForOp = cast<OpTy>(rootForOp.getBody()->front());
    }
  }
  return rootForOp;
}

// Replace the rooted at "rootForOp" with a GPU launch operation.  This expects
// "innermostForOp" to point to the last loop to be transformed to the kernel,
// and to have (numBlockDims + numThreadDims) perfectly nested loops between
// "rootForOp" and "innermostForOp".
// TODO(ravishankarm) : This method can be modified to use the
// createLaunchFromOp method, since that is a strict generalization of this
// method.
template <typename OpTy>
void LoopToGpuConverter::createLaunch(OpTy rootForOp, OpTy innermostForOp,
                                      unsigned numBlockDims,
                                      unsigned numThreadDims) {
  OpBuilder builder(rootForOp.getOperation());
  // Prepare the grid and block sizes for the launch operation.  If there is
  // no loop mapped to a specific dimension, use constant "1" as its size.
  Value constOne = (numBlockDims < 3 || numThreadDims < 3)
                       ? builder.create<ConstantIndexOp>(rootForOp.getLoc(), 1)
                       : nullptr;
  Value gridSizeX = numBlockDims > 0 ? dims[0] : constOne;
  Value gridSizeY = numBlockDims > 1 ? dims[1] : constOne;
  Value gridSizeZ = numBlockDims > 2 ? dims[2] : constOne;
  Value blockSizeX = numThreadDims > 0 ? dims[numBlockDims] : constOne;
  Value blockSizeY = numThreadDims > 1 ? dims[numBlockDims + 1] : constOne;
  Value blockSizeZ = numThreadDims > 2 ? dims[numBlockDims + 2] : constOne;

  // Create a launch op and move the body region of the innermost loop to the
  // launch op.
  auto launchOp = builder.create<gpu::LaunchOp>(
      rootForOp.getLoc(), gridSizeX, gridSizeY, gridSizeZ, blockSizeX,
      blockSizeY, blockSizeZ);

  // Replace the loop terminator (loops contain only a single block) with the
  // gpu terminator and move the operations from the loop body block to the gpu
  // launch body block.  Do not move the entire block because of the difference
  // in block arguments.
  Operation &terminator = innermostForOp.getBody()->back();
  Location terminatorLoc = terminator.getLoc();
  terminator.erase();
  builder.setInsertionPointToEnd(innermostForOp.getBody());
  builder.create<gpu::TerminatorOp>(terminatorLoc, llvm::None);
  launchOp.body().front().getOperations().splice(
      launchOp.body().front().begin(),
      innermostForOp.getBody()->getOperations());

  // Remap the loop iterators to use block/thread identifiers instead.  Loops
  // may iterate from LB with step S whereas GPU thread/block ids always iterate
  // from 0 to N with step 1.  Therefore, loop induction variables are replaced
  // with (gpu-thread/block-id * S) + LB.
  builder.setInsertionPointToStart(&launchOp.body().front());
  auto lbArgumentIt = lbs.begin();
  auto stepArgumentIt = steps.begin();
  for (auto en : llvm::enumerate(ivs)) {
    Value id =
        en.index() < numBlockDims
            ? getDim3Value(launchOp.getBlockIds(), en.index())
            : getDim3Value(launchOp.getThreadIds(), en.index() - numBlockDims);
    Value step = steps[en.index()];
    if (!isConstantOne(step))
      id = builder.create<MulIOp>(rootForOp.getLoc(), step, id);

    Value ivReplacement =
        builder.create<AddIOp>(rootForOp.getLoc(), *lbArgumentIt, id);
    en.value().replaceAllUsesWith(ivReplacement);
    std::advance(lbArgumentIt, 1);
    std::advance(stepArgumentIt, 1);
  }

  // We are done and can erase the original outermost loop.
  rootForOp.erase();
}

// Generic loop to GPU kernel conversion function.
template <typename OpTy>
static LogicalResult convertLoopNestToGPULaunch(OpTy forOp,
                                                unsigned numBlockDims,
                                                unsigned numThreadDims) {
  if (failed(checkLoopNestMappable(forOp, numBlockDims, numThreadDims)))
    return failure();

  LoopToGpuConverter converter;
  auto maybeInnerLoop =
      converter.collectBounds(forOp, numBlockDims + numThreadDims);
  if (!maybeInnerLoop)
    return failure();
  converter.createLaunch(forOp, *maybeInnerLoop, numBlockDims, numThreadDims);

  return success();
}

namespace pmlc::conversion::scf_to_gpu {

LogicalResult convertAffineLoopNestToGPULaunch(AffineForOp forOp,
                                               unsigned numBlockDims,
                                               unsigned numThreadDims) {
  return ::convertLoopNestToGPULaunch(forOp, numBlockDims, numThreadDims);
}

LogicalResult convertLoopNestToGPULaunch(ForOp forOp, unsigned numBlockDims,
                                         unsigned numThreadDims) {
  return ::convertLoopNestToGPULaunch(forOp, numBlockDims, numThreadDims);
}

} // namespace pmlc::conversion::scf_to_gpu
