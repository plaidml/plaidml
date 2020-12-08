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

#include <unordered_map>
#include <unordered_set>

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "pmlc/transforms/pass_detail.h"
#include "pmlc/util/loop_with_epilog.h"

#include "llvm/ADT/TypeSwitch.h"
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
// An ideal implementation might scan the operation list, identifying sets of
// operations to be hoisted to the prolog and epilog as a unit.  For each
// operation, we might start by assuming that it's hoistable, and then filter
// out unhoistable operations:
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
// * A free can only be hoisted if the corresponding alloc is hoisted.
//
//   * When a free operation is hoisted, if the value is used within the loop,
//     it must be hoisted to the epilog; otherwise, it should be hoisted to the
//     prolog.  (NB Currently, LoopLikeOpInterface only supports moving
//     operations to the prolog; a free operation that needs to be hoisted to
//     the epilog will require an interface extension to be hoistable at all).
//
//   * Note that a freeing operation might have other side-effects.  So hoisting
//     a free to the prolog requires a different conflict analysis than
//     hoisting a free to the epilog.  This rapidly becomes an interesting
//     problem to solve in the general case.
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
//
// * An operation that has recursive side effects has the logical effects
//   and dependencies of its nested operations.
//
//
// The current implementation is not ideal, since getting reads and writes
// correct in the general case is a little tricky.  Instead, we use the
// following simple algorithm, which works well for the cases we're
// interested in:
//
// for op in all_operations_in_the_loop:
//   If any input is defined within the loop, we don't hoist.
//   If the operation has no side-effects or recursive side-effects, hoist it.
//   If the operation reads values written within the loop, skip hoisting.
//   If the operation writes values written by another operation within the
//     loop, skip hoisting.
//   If the operation allocs a value, and the corresponding free is hoistable
//     by the above rules and has no recursive side effects, hoist both as a
//     unit, with the free going to the epilog.
//   If an allocation allocs multiple values, it's not hoistable.
//   If the operation has recursive side effects, check that all of its
//     nested operations are hoistable by the above rules.

class HoistingPass final : public HoistingPassBase<HoistingPass> {
public:
  void runOnOperation() final;

private:
  LogicalResult hoistOps(LoopLikeOpInterface loopOp);
};

static LogicalResult
verifyNoBodyWritersExcept(Value value, Operation *op,
                          function_ref<bool(Operation *)> opOutside) {
  for (auto &use : value.getUses()) {
    Operation *userOp = use.getOwner();
    auto userMemInterface = dyn_cast<MemoryEffectOpInterface>(userOp);
    if (!userMemInterface) {
      // This user isn't declaring how they use the value.
      // Conservatively, we fail the check.
      return failure();
    }
    SmallVector<MemoryEffects::EffectInstance, 2> userEffects;
    userMemInterface.getEffectsOnValue(value, userEffects);
    for (auto &userEffectInstance : userEffects) {
      auto *userEffect = userEffectInstance.getEffect();
      if (userEffect && isa<MemoryEffects::Free, MemoryEffects::Allocate,
                            MemoryEffects::Write>(userEffect)) {
        // Make sure that this use is outside the loop.
        if (!opOutside(userOp)) {
          return failure();
        }
      }
    }
  }
  return success();
}

using HoistResult = llvm::PointerIntPair<Operation *, 1, bool>;

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
static HoistResult
canBeHoistedToProlog(Operation *op, bool allowAllocs,
                     function_ref<bool(Value)> definedOutside,
                     function_ref<bool(Operation *)> opOutside) {
  // Check that dependencies are defined outside of loop.
  if (!llvm::all_of(op->getOperands(), definedOutside)) {
    return HoistResult{nullptr, false};
  }

  Operation *epilogOp = nullptr;

  // Check whether this op is side-effect free. If we already know that there
  // can be no side-effects because the surrounding op has claimed so, we can
  // (and have to) skip this step.
  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance, 8> effectInstances;
    memInterface.getEffects(effectInstances);
    for (auto &effectInstance : effectInstances) {
      auto result =
          TypeSwitch<MemoryEffects::Effect *, LogicalResult>(
              effectInstance.getEffect())
              .Case<MemoryEffects::Read>([&](MemoryEffects::Read * /* read */) {
                // In order to hoist this operation, we need to check that all
                // writes to the value have been hoisted out of the loop.
                // (Technically, we could be more lenient, but this check
                // suffices for now.)
                return verifyNoBodyWritersExcept(effectInstance.getValue(),
                                                 nullptr, opOutside);
              })
              .Case<MemoryEffects::Write>(
                  [&](MemoryEffects::Write * /* write */) {
                    // In order to hoist this operation, we need to check that
                    // all other writes to the value have been hoisted out of
                    // the loop. (Technically, we could be more lenient, but
                    // this check suffices for now.)
                    return verifyNoBodyWritersExcept(effectInstance.getValue(),
                                                     op, opOutside);
                  })
              .Case<MemoryEffects::Allocate>([&](MemoryEffects::Allocate
                                                     * /* alloc */) {
                if (!allowAllocs) {
                  return failure();
                }
                if (epilogOp) {
                  // We already have a free for this operation!  We don't
                  // currently support multiple allocations from a single
                  // operation.
                  return failure();
                }
                auto value = effectInstance.getValue();
                if (!value) {
                  // Unhoistable, since we can't pair this allocation
                  // with a corresponding free.
                  return failure();
                }
                if (value.getDefiningOp() != op) {
                  // We only hoist allocs where the allocating operation
                  // is the operation producing the value.
                  return failure();
                }
                Operation *freeOp = nullptr;
                MemoryEffectOpInterface freeOpMemEffectInterface;
                for (auto &use : value.getUses()) {
                  Operation *userOp = use.getOwner();
                  auto userMemInterface =
                      dyn_cast<MemoryEffectOpInterface>(userOp);
                  if (!userMemInterface) {
                    // This user doesn't provide memory effect information.
                    // Since the allocation provided memory effect
                    // information, we'll assume that this use is a read or
                    // a write, not the free that we're looking for.
                    continue;
                  }
                  SmallVector<MemoryEffects::EffectInstance, 2> userEffects;
                  userMemInterface.getEffectsOnValue(value, userEffects);
                  for (auto &userEffectInstance : userEffects) {
                    auto *userEffect = userEffectInstance.getEffect();
                    if (userEffect && isa<MemoryEffects::Free>(userEffect)) {
                      // This is the freeing operation.
                      if (freeOp) {
                        return failure();
                      }
                      freeOp = userOp;
                      freeOpMemEffectInterface = userMemInterface;
                    }
                  }
                }
                if (!freeOp) {
                  // The value was not declared to be freed.
                  return failure();
                }
                // Validate that the free operation can be hoisted.
                if (!llvm::all_of(freeOp->getOperands(),
                                  [&](Value freeOperand) {
                                    return freeOperand == value ||
                                           definedOutside(freeOperand);
                                  })) {
                  return failure();
                }
                if (freeOp->getNumRegions() != 0) {
                  // We don't hoist free operations with regions.
                  return failure();
                }
                SmallVector<MemoryEffects::EffectInstance, 8>
                    freeEffectInstances;
                freeOpMemEffectInterface.getEffects(freeEffectInstances);
                for (auto &freeEffectInstance : freeEffectInstances) {
                  auto result =
                      TypeSwitch<MemoryEffects::Effect *, LogicalResult>(
                          freeEffectInstance.getEffect())
                          .Case<MemoryEffects::Read>(
                              [&](MemoryEffects::Read * /* read */) {
                                return verifyNoBodyWritersExcept(
                                    freeEffectInstance.getValue(), nullptr,
                                    opOutside);
                              })
                          .Case<MemoryEffects::Write>(
                              [&](MemoryEffects::Write * /* write */) {
                                // We don't hoist free operations that write
                                // anything.
                                return failure();
                              })
                          .Case<MemoryEffects::Allocate>(
                              [&](MemoryEffects::Allocate * /* allocate */) {
                                // We don't hoist free operations that allocate
                                // anything.
                                return failure();
                              })
                          .Case<MemoryEffects::Free>(
                              [&](MemoryEffects::Free * /* free */) {
                                if (freeEffectInstance.getValue() == value) {
                                  return success();
                                }
                                // Otherwise, this operation is also freeing
                                // something else. We could possibly handle this
                                // in the future, but not at present.
                                return failure();
                              })
                          .Default([&](MemoryEffects::Effect * /* effect */) {
                            return failure();
                          });
                  if (failed(result)) {
                    return failure();
                  }
                }
                epilogOp = freeOp;
                return success();
              })
              .Case<MemoryEffects::Free>([&](MemoryEffects::Free * /* free */) {
                // We never hoist free ops here; they're filtered out (due to
                // already being hoisted) in hoistOps() before this function
                // is called.
                return failure();
              })
              .Default([&](MemoryEffects::Effect * /* effect */) {
                return failure();
              });
      if (failed(result)) {
        return HoistResult{nullptr, false};
      }
    }

    if (!op->hasTrait<OpTrait::HasRecursiveSideEffects>()) {
      // If the operation doesn't have side effects and it doesn't recursively
      // have side effects, it can always be hoisted.
      return HoistResult{epilogOp, true};
    }

  } else if (!op->hasTrait<OpTrait::HasRecursiveSideEffects>()) {
    // Otherwise, if the operation doesn't provide the memory effect interface
    // and it doesn't have recursive side effects, we treat it conservatively as
    // side-effecting.
    return HoistResult{nullptr, false};
  }

  // Recurse into the regions for this op and check whether the contained ops
  // can be hoisted.
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      for (auto &innerOp : block.without_terminator()) {
        if (!canBeHoistedToProlog(&innerOp, allowAllocs, definedOutside,
                                  opOutside)
                 .getInt()) {
          return HoistResult{nullptr, false};
        }
      }
    }
  }
  return HoistResult{epilogOp, true};
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

LogicalResult HoistingPass::hoistOps(LoopLikeOpInterface loopOp) {
  auto loopWithEpilog =
      dyn_cast<util::LoopWithEpilogInterface>(loopOp.getOperation());
  bool hasLoopWithEpilog = loopWithEpilog;
  auto &loopBody = loopOp.getLoopBody();

  // The set of operations to be moved before the loop (making their
  // results available to operations within the loop).
  SmallPtrSet<Operation *, 8> willBeMovedToPrologSet;

  // The set of operations to be moved before or after the loop.
  SmallPtrSet<Operation *, 8> willBeMovedSet;

  // The list of operations to be moved before the loop, in order
  // (we preserve the existing order to ensure IR sanity).
  SmallVector<Operation *, 8> opsToMoveToProlog;

  // The list of operations to be moved after the loop, in order
  // of allocations (reversed to insertion order just before passing to the
  // loop).
  SmallVector<Operation *, 8> opsToMoveToEpilog;

  // Helper to check whether an operation is loop invariant wrt. SSA properties.
  auto isDefinedOutsideOfBody = [&](Value value) {
    auto definingOp = value.getDefiningOp();
    return (definingOp && !!willBeMovedToPrologSet.count(definingOp)) ||
           loopOp.isDefinedOutsideOfLoop(value);
  };

  // Helper to check whether an operation is in the loop body, taking into
  // account operations scheduled to be hoisted out of the loop body.
  auto isOpOutsideOfBody = [&](Operation *op) {
    while (op) {
      if (op->getParentRegion() == &loopBody) {
        if (willBeMovedSet.count(op)) {
          return true;
        }
        return false;
      }
      op = op->getParentOp();
    }
    return true;
  };

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there. These regions might have semantics unknown to this
  // rewriting. If the nested regions are loops, they will have been processed.
  for (auto &block : loopBody) {
    for (auto &op : block.without_terminator()) {
      if (willBeMovedSet.count(&op)) {
        // Presumably, this operation is being moved to the epilog.
        continue;
      }
      auto result =
          canBeHoistedToProlog(&op, /*allowAllocs=*/hasLoopWithEpilog,
                               isDefinedOutsideOfBody, isOpOutsideOfBody);
      if (result.getInt()) {
        opsToMoveToProlog.emplace_back(&op);
        willBeMovedToPrologSet.insert(&op);
        willBeMovedSet.insert(&op);
        if (result.getPointer()) {
          opsToMoveToEpilog.emplace_back(result.getPointer());
          willBeMovedSet.insert(result.getPointer());
        }
      }
    }
  }

  // For all instructions that we found to be invariant, move outside of the
  // loop.
  auto result = loopOp.moveOutOfLoop(opsToMoveToProlog);
  if (succeeded(result) && loopWithEpilog) {
    std::reverse(opsToMoveToEpilog.begin(), opsToMoveToEpilog.end());
    result = loopWithEpilog.moveToLoopEpilog(opsToMoveToEpilog);
  }
  LLVM_DEBUG(loopOp.print(llvm::dbgs() << "\n\nModified loop:\n"));
  return result;
}

} // namespace

std::unique_ptr<Pass> createHoistingPass() {
  return std::make_unique<HoistingPass>();
}

} // namespace pmlc::transforms
