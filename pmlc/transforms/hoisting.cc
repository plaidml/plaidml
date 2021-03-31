// Copyright 2020 Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/transforms/pass_detail.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/memuse.h"

using namespace mlir;                // NOLINT
using namespace pmlc::dialect::stdx; // NOLINT

namespace pmlc::transforms {
namespace {

// Per module state for hoisting
class HoistingState {
public:
  explicit HoistingState(ModuleOp op) : op(op) {}

  // Check if this module matches the 'init/main/fini' protocol, and also set
  // internal members if so.
  bool matchesProtocol();

  // Wire up the definitions packed and returned by init pack directly into
  // their uses from the main/fini unpack.  This temporarily makes use/def
  // relations across function to allow hoisting to be easier, and is undone in
  // disconnectFunctions()
  void connectFunctions();

  // Move operations from main -> init/fini as approriate
  void doHoisting();

  // Find the set of values actually moving across function boundaries post
  // hoisting, and disconnect the use-def chains via rewriting pack/unpack
  void disconnectFunctions();

private:
  // Main module op
  ModuleOp op;
  // Protocol functions
  FuncOp initFunc;
  FuncOp mainFunc;
  FuncOp finiFunc;
  // Associated op
  PackOp initPack;
  UnpackOp mainUnpack;
  UnpackOp finiUnpack;
  // List of values to consider as boundry crossing in disconnect
  DenseSet<Value> inInit;
};

bool HoistingState::matchesProtocol() {
  // Find init/main/fini, if not found, skip optimizations
  initFunc = op.lookupSymbol<FuncOp>("init");
  mainFunc = op.lookupSymbol<FuncOp>("main");
  finiFunc = op.lookupSymbol<FuncOp>("fini");
  if (!initFunc || !mainFunc || !finiFunc) {
    return false;
  }
  // Get + verify pack/unpack, fail early if preconditions are not met
  auto initRet = dyn_cast<ReturnOp>(initFunc.front().getTerminator());
  if (!initRet || initRet.getNumOperands() != 1)
    return false;
  if (mainFunc.getNumArguments() < 1 || !mainFunc.getArgument(0).hasOneUse())
    return false;
  if (finiFunc.getNumArguments() < 1 || !finiFunc.getArgument(0).hasOneUse())
    return false;
  initPack = dyn_cast_or_null<PackOp>(initRet.getOperand(0).getDefiningOp());
  mainUnpack = dyn_cast<UnpackOp>(*mainFunc.getArgument(0).user_begin());
  finiUnpack = dyn_cast<UnpackOp>(*finiFunc.getArgument(0).user_begin());
  if (!initPack || !mainUnpack || !finiUnpack) {
    return false;
  }
  // Verify typing matches
  if (initPack.getOperandTypes() != mainUnpack.getResultTypes() ||
      initPack.getOperandTypes() != finiUnpack.getResultTypes()) {
    return false;
  }
  // All is well
  return true;
}

void HoistingState::connectFunctions() {
  // 'Disolve' function boundaries by forwarding values temporarily
  // Mark forwared values as possible final cross function values
  for (unsigned i = 0; i < initPack.getNumOperands(); i++) {
    mainUnpack.getResult(i).replaceAllUsesWith(initPack.getOperand(i));
    finiUnpack.getResult(i).replaceAllUsesWith(initPack.getOperand(i));
    inInit.insert(initPack.getOperand(i));
  }
}

void HoistingState::doHoisting() {
  // Go over every op on main:
  // If it's (no-side-effect or an alloc/dealloc pair) and all of it's operands
  // are defined in init, hoist.
  for (auto &innerOp :
       make_early_inc_range(mainFunc.begin()->without_terminator())) {
    // If an operation has a side effect, bail
    IVLOG(3, "Trying op " << debugString(innerOp));
    // Skip constant ops
    if (isa<pmlc::dialect::tile::ConstantOp>(innerOp))
      continue;
    // If we dom't have memory effect interface, we can't hoist
    auto innerEffect = dyn_cast<MemoryEffectOpInterface>(innerOp);
    if (!innerEffect)
      continue;
    // Check for hoistable cases
    bool validToHoist = false;
    // See if we are an alloc for an alloc/dealloc pair, or if we are
    // no-side-effect.
    Operation *dealloc = util::findDeallocPair(&innerOp);
    if (dealloc && dealloc->getBlock() == innerOp.getBlock()) {
      validToHoist = true;
    }
    if (innerEffect.hasNoEffect()) {
      validToHoist = true;
    }
    if (!validToHoist) {
      continue;
    }
    // Check if all operands (if any) are in init (or constant)
    IVLOG(3, "Checking operands " << debugString(innerOp));
    bool allOperandsInInit = true;
    DenseSet<mlir::Operation *> constantsToCopy;
    for (auto operand : innerOp.getOperands()) {
      if (operand.getDefiningOp() &&
          isa<pmlc::dialect::tile::ConstantOp>(operand.getDefiningOp())) {
        if (!constantsToCopy.count(operand.getDefiningOp()))
          constantsToCopy.insert(operand.getDefiningOp());
      } else if (!inInit.count(operand)) {
        allOperandsInInit = false;
        break;
      }
    }
    if (!allOperandsInInit) {
      continue;
    }
    IVLOG(3, "Hoisting " << debugString(innerOp));
    // Yes?  Hoist to init, add results as possible cross function results
    innerOp.moveBefore(initPack);
    // If we have a 'dealloc', move that into fini
    if (dealloc) {
      dealloc->moveAfter(finiUnpack);
    }
    // Copy the constant operands into init
    for (auto constantOp : constantsToCopy) {
      OpBuilder builder(innerOp.getBlock(), innerOp.getIterator());
      auto newConstOp = builder.clone(*constantOp);
      for (size_t i = 0; i < innerOp.getOperands().size(); i++) {
        if (innerOp.getOperand(i).getDefiningOp() == constantOp)
          innerOp.setOperand(i, newConstOp->getResult(0));
      }
    }
    for (auto result : innerOp.getResults()) {
      inInit.insert(result);
    }
  }
}

void HoistingState::disconnectFunctions() {
  // Go over all possible cross function values and prep for reconnecting.
  SmallVector<Value, 8> packVals;
  SmallVector<Type, 8> packTypes;
  DenseMap<Value, unsigned> toIndex;
  SmallVector<OpOperand *, 8> mainUnpackUses;
  SmallVector<OpOperand *, 8> finiUnpackUses;
  for (auto val : inInit) {
    bool crossFunc = false;
    for (auto &use : val.getUses()) {
      if (mainFunc.getOperation()->isAncestor(use.getOwner())) {
        crossFunc = true;
        mainUnpackUses.push_back(&use);
      }
      if (finiFunc.getOperation()->isAncestor(use.getOwner())) {
        crossFunc = true;
        finiUnpackUses.push_back(&use);
      }
    }
    if (crossFunc) {
      toIndex[val] = packVals.size();
      packVals.push_back(val);
      packTypes.push_back(val.getType());
    }
  }
  // Reconstruct pack/unpack
  // Replace initPack
  OpBuilder builder(initPack);
  auto argpackType = ArgpackType::get(op.getContext());
  auto newPack = builder.create<PackOp>(initPack.getLoc(),
                                        TypeRange(argpackType), packVals);
  initPack.getResult().replaceAllUsesWith(newPack);
  initPack.erase();
  // Replace mainUnpack
  builder.setInsertionPointToStart(&mainFunc.front());
  auto newMainUnpack =
      builder.create<UnpackOp>(mainUnpack.getLoc(), packTypes, mainUnpack.in());
  for (auto use : mainUnpackUses) {
    use->set(newMainUnpack.getResult(toIndex[use->get()]));
  }
  mainUnpack.erase();
  // Replace finiUnpack
  builder.setInsertionPointToStart(&finiFunc.front());
  auto newFiniUnpack =
      builder.create<UnpackOp>(finiUnpack.getLoc(), packTypes, finiUnpack.in());
  for (auto use : finiUnpackUses) {
    use->set(newFiniUnpack.getResult(toIndex[use->get()]));
  }
  finiUnpack.erase();
}

class HoistingPass final : public HoistingPassBase<HoistingPass> {
public:
  void runOnOperation() final;
};

void HoistingPass::runOnOperation() {
  ModuleOp op = getOperation();
  IVLOG(3, "Hoisting");
  HoistingState state(op);
  if (!state.matchesProtocol()) {
    IVLOG(3, "Failed to match protocl");
    return;
  }
  state.connectFunctions();
  IVLOG(3, "Doing it");
  state.doHoisting();
  state.disconnectFunctions();
}

} // namespace

std::unique_ptr<Pass> createHoistingPass() {
  return std::make_unique<HoistingPass>();
}

} // namespace pmlc::transforms
