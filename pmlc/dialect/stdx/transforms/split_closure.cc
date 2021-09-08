// Copyright 2021 Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"

#include "pmlc/dialect/stdx/transforms/pass_detail.h"
#include "pmlc/dialect/stdx/transforms/passes.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::stdx {

namespace {

struct SplitClosurePass : public SplitClosureBase<SplitClosurePass> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    FuncOp main = module.lookupSymbol<FuncOp>("main");
    if (!main)
      return;

    auto it = main.body().op_begin<ClosureOp>();
    if (it == main.body().op_end<ClosureOp>())
      return;

    splitClosure(*it);
  }

  void splitClosure(ClosureOp op) {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    FuncOp func = op->getParentOfType<FuncOp>();
    auto &funcOps = func.body().front().getOperations();
    auto itOp = Block::iterator(op);
    auto itNextOp = std::next(itOp);

    SetVector<Value> values;
    getUsedValuesDefinedAbove(op.body(), op.body(), values);

    Block *cleanup = func.body().front().splitBlock(itNextOp);
    getUsedValuesDefinedOutside(cleanup, values);

    SmallVector<Type> packedTypes;
    for (Value value : values) {
      packedTypes.push_back(value.getType());
    }

    auto tupleType = TupleType::get(context, packedTypes);
    auto initFuncType =
        FunctionType::get(context, func.getType().getInputs(), {tupleType});
    auto mainFuncType = FunctionType::get(context, op.getType().getInputs(),
                                          func.getType().getResults());
    auto finiFuncType = FunctionType::get(context, {tupleType}, {});

    ImplicitLocOpBuilder builder(op.getLoc(), func);
    auto init = builder.create<FuncOp>("init", initFuncType);
    auto main = builder.create<FuncOp>("main", mainFuncType);
    auto fini = builder.create<FuncOp>("fini", finiFuncType);

    // Construct the `init` function.
    builder.setInsertionPointToStart(init.addEntryBlock());
    auto packOp = builder.create<PackOp>(tupleType, values.getArrayRef());
    builder.create<ReturnOp>(packOp.getResult());
    auto &initOps = init.body().front().getOperations();
    initOps.splice(initOps.begin(), funcOps, funcOps.begin(), itOp);

    // Construct the new `main` function.
    main.body().takeBody(op.body());
    Operation *yield = main.body().front().getTerminator();
    builder.setInsertionPoint(yield);
    builder.create<ReturnOp>();
    yield->erase();

    main.insertArgument(0, tupleType, /*argAttrs=*/nullptr);
    builder.setInsertionPointToStart(&main.body().front());
    auto mainUnpackOp =
        builder.create<UnpackOp>(packedTypes, main.getArgument(0));
    replaceWithUnpacked(values.getArrayRef(), mainUnpackOp, main);

    // Construct the `fini` function.
    builder.setInsertionPointToStart(fini.addEntryBlock());
    auto finiUnpackOp =
        builder.create<UnpackOp>(packedTypes, fini.getArgument(0));
    auto &finiOps = fini.body().front().getOperations();
    finiOps.splice(finiOps.end(), funcOps, cleanup->begin(), cleanup->end());
    replaceWithUnpacked(values.getArrayRef(), finiUnpackOp, fini);

    // NOTE: we need to replace these at the end so that the `values` used in
    // `replaceWithUnpacked` remain valid.
    for (BlockArgument arg : func.getArguments()) {
      arg.replaceAllUsesWith(init.getArgument(arg.getArgNumber()));
    }

    func.erase();
  }

  void replaceWithUnpacked(ArrayRef<Value> values, UnpackOp unpackOp,
                           FuncOp func) {
    for (auto it : llvm::enumerate(values)) {
      Value value = it.value();
      value.replaceUsesWithIf(
          unpackOp.getResult(it.index()), [&](OpOperand &operand) {
            if (!operand.getOwner())
              return false;
            return operand.getOwner()->getParentOfType<FuncOp>() == func;
          });
    }
  }

  void getUsedValuesDefinedOutside(Block *block, SetVector<Value> &values) {
    block->walk([&](Operation *op) {
      for (OpOperand &operand : op->getOpOperands()) {
        if (operand.get().getParentBlock() != block)
          values.insert(operand.get());
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createSplitClosurePass() {
  return std::make_unique<SplitClosurePass>();
}

} // namespace pmlc::dialect::stdx
