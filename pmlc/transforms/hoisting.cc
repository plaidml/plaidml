// PlaidML operation hoisting, originally from the LLVM project, and
// subsequently modified by Intel Corporation.
//
// Original copyright:
//
//===- LoopInvariantCodeMotion.cpp - Code to perform loop fusion-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "pmlc/transforms/pass_detail.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hoist"

using namespace mlir; // NOLINT

namespace pmlc::transforms {
namespace {

// The hoisting pass moves operations before and after loops, hopefully
// reducing the per-iteration latency while preserving the same logical
// semantics.
//
// Conceptually (without actually realizing the result), we unroll the loop,
// looking for operations that can be moved to the beginning (prolog) or the end
// (epilog) of the unrolled sequence without changing the memory access
// semantics.
//
// More specifically, we scan the operation list, identifying sets of operations
// to be hoisted as a unit.  For each operation, we start by assuming that it's
// hoistable, and then filter out unhoistable operations:
//
// * Operations cannot be hoisted if they depend on values defined within the
//   loop.
//
// * Operations with recursive side-effects cannot be hoisted unless every
//   instruction in their regions is hoistable (in sequence, per-region).
//
// * An allocating operation cannot be hoisted unless the corresponding free
//   is hoisted.
//
//     * We assume that allocation also performs a write of unknown data.
//       (Some allocation instructions may also write known data; these should
//       typically be separated into independent alloc and write ops if
//       possible, so that they can be hoisted independently.)
//
//     * => We assume that it's legal to elide a free/alloc pair (as long as
//       we remember that this was logically writing the memory contents, which
//       is important when we're analyzing write and read operations).
//
//     * => It's legal to pair up a free at the end of a loop with an alloc
//       at the beginning of the loop, logically elide them, and then hoist
//       the alloc at the front of the conceptually unrolled loop together
//       with the free at the end of the conceptually unrolled loop.
//
// * A free can only be hoisted if the corresponding alloc is hoisted; since
//   we match these up when we consider alloc operations, we don't separately
//   hoist free operations.
//
//   * When a free operation is hoisted, if the value is used within the loop,
//     it must be hoisted to the epilog; otherwise, it should be hoisted to the
//     prolog.  (NB Currently, LoopLikeOpInterface only supports moving
//     operations to the prolog; a free operation that needs to be hoisted to
//     the epilog will require an interface extension to be hoistable at all).
//
// * A read of a value cannot be hoisted unless the previous write to the value
//   is also hoisted (and note that the write may occur in a different loop
//   iteration, unless there was originally a free/alloc sequence creating
//   a new value per iteration).
//
// * A write of a value cannot be hoisted unless all reads between the write
//   and the next write to the value are also hoisted (and note that the reads
//   may occur in a different loop iteration, unless there was originally a
//   free/alloc sequence creating a new value per iteration).
//
// * An operation that both reads and writes a value is presumed to read and
//   then write.  Whether such an operation is hoistable depends on whether
//   the value originally crossed loop iterations.

class HoistingPass final : public HoistingPassBase<HoistingPass> {
public:
  void runOnOperation() final;

private:
  LogicalResult hoistOps(LoopLikeOpInterface loopOp);
};

// Checks whether the given op can be hoisted by checking that
// - the op and any of its contained operations do not depend on SSA values
//   defined inside of the loop (by means of calling definedOutside).
// - the op has no un-hoistable side-effects.
//
// The result is a pair: a boolean that's true iff the operation can be
// hoisted before the loop, and an Operation* that, if non-nullptr, indicates
// a paired operation that must be hoisted after the loop when hoisting the
// supplied operation.  (This is used for hoisting allocation/deallocation
// pairs.)
static bool canBeHoisted(Operation *op,
                         function_ref<bool(Value)> definedOutside) {
  // Check that dependencies are defined outside of loop.
  if (!llvm::all_of(op->getOperands(), definedOutside)) {
    return false;
  }

  // Check whether this op is side-effect free. If we already know that there
  // can be no side-effects because the surrounding op has claimed so, we can
  // (and have to) skip this step.
  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    if (!memInterface.hasNoEffect()) {
      // Non-immediate children need to have no memory effect in order to be
      // hoistable.
      return false;
    }

    if (!op->hasTrait<OpTrait::HasRecursiveSideEffects>()) {
      // If the operation doesn't have side effects and it doesn't recursively
      // have side effects, it can always be hoisted.
      return true;
    }

  } else if (!op->hasTrait<OpTrait::HasRecursiveSideEffects>()) {
    // Otherwise, if the operation doesn't provide the memory effect interface
    // and it doesn't have recursive side effects, we treat it conservatively as
    // side-effecting.
    return false;
  }

  // Recurse into the regions for this op and check whether the contained ops
  // can be hoisted.
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      for (auto &innerOp : block.without_terminator()) {
        if (!canBeHoisted(&innerOp, definedOutside)) {
          return false;
        }
      }
    }
  }
  return true;
}

void HoistingPass::runOnOperation() {
  // Process all loops, in innermost-loop-first order, hoisting as we go.
  getOperation().walk([this](LoopLikeOpInterface op) {
    if (failed(hoistOps(op))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

LogicalResult HoistingPass::hoistOps(LoopLikeOpInterface looplike) {
  auto &loopBody = looplike.getLoopBody();

  // The set of operations to be moved before the loop (making their
  // results available to operations within the loop).
  SmallPtrSet<Operation *, 8> willBeMovedSet;

  // The list of operations to be moved before the loop, in order
  // (we preserve the existing order to ensure IR sanity).
  SmallVector<Operation *, 8> opsToMove;

  // Helper to check whether an operation is loop invariant wrt. SSA properties.
  auto isDefinedOutsideOfBody = [&](Value value) {
    auto definingOp = value.getDefiningOp();
    return (definingOp && !!willBeMovedSet.count(definingOp)) ||
           looplike.isDefinedOutsideOfLoop(value);
  };

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there. These regions might have semantics unknown to this
  // rewriting. If the nested regions are loops, they will have been processed.
  for (auto &block : loopBody) {
    for (auto &op : block.without_terminator()) {
      if (willBeMovedSet.count(&op)) {
        // We've already decided to hoist this op (=> it must be a hoist-before
        // write to a value produced by a hoist-before operation, or it must be
        // a hoist-after operation paired with an earlier hoist-before
        // operation).
        continue;
      }
      if (canBeHoisted(&op, isDefinedOutsideOfBody)) {
        opsToMove.emplace_back(&op);
        willBeMovedSet.insert(&op);
      }
    }
  }

  // For all instructions that we found to be invariant, move outside of the
  // loop.
  auto result = looplike.moveOutOfLoop(opsToMove);
  LLVM_DEBUG(looplike.print(llvm::dbgs() << "\n\nModified loop:\n"));
  return result;
}

} // namespace

std::unique_ptr<Pass> createHoistingPass() {
  return std::make_unique<HoistingPass>();
}

} // namespace pmlc::transforms
