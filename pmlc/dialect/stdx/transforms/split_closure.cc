// Copyright 2021 Intel Corporation

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "pmlc/dialect/stdx/transforms/pass_detail.h"
#include "pmlc/dialect/stdx/transforms/passes.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::stdx {

namespace {

struct ValuesWithCast {
  Value value;
  memref::CastOp castOp;
};

struct SplitClosurePass : public SplitClosureBase<SplitClosurePass> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    func::FuncOp main = module.lookupSymbol<func::FuncOp>("main");
    if (!main)
      return;

    auto it = main.getBody().op_begin<ClosureOp>();
    if (it == main.getBody().op_end<ClosureOp>())
      return;

    splitClosure(*it);
  }

  void splitClosure(ClosureOp op) {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    func::FuncOp func = op->getParentOfType<func::FuncOp>();
    auto &funcOps = func.getBody().front().getOperations();
    auto itOp = Block::iterator(op);
    auto itNextOp = std::next(itOp);

    SetVector<Value> values;
    visitUsedValuesDefinedAbove(op.body(), op.body(), [&](OpOperand *operand) {
      addUsedValue(*operand, values);
    });

    Block *cleanup = func.getBody().front().splitBlock(itNextOp);
    getUsedValuesDefinedOutside(cleanup, values);

    SmallVector<Type> packedTypes;
    SmallVector<Value> packedValues;
    SmallVector<ValuesWithCast> valuesWithCast;
    for (Value value : values) {
      if (auto op = dyn_cast_or_null<memref::CastOp>(value.getDefiningOp())) {
        packedValues.push_back(op.source());
        packedTypes.push_back(op.source().getType());
        valuesWithCast.emplace_back(ValuesWithCast{value, op});
      } else {
        packedValues.push_back(value);
        packedTypes.push_back(value.getType());
        valuesWithCast.emplace_back(ValuesWithCast{value, nullptr});
      }
    }

    auto tupleType = TupleType::get(context, packedTypes);
    auto initFuncType =
        FunctionType::get(context, func.getFunctionType().getInputs(), {tupleType});
    auto mainFuncType = FunctionType::get(context, op.getFunctionType().getInputs(),
                                          func.getFunctionType().getResults());
    auto finiFuncType = FunctionType::get(context, {tupleType}, {});

    ImplicitLocOpBuilder builder(op.getLoc(), func);
    auto init = builder.create<func::FuncOp>("init", initFuncType);
    auto main = builder.create<func::FuncOp>("main", mainFuncType);
    auto fini = builder.create<func::FuncOp>("fini", finiFuncType);

    // Construct the `init` function.
    builder.setInsertionPointToStart(init.addEntryBlock());
    auto packOp = builder.create<PackOp>(tupleType, packedValues);
    builder.create<func::ReturnOp>(packOp.getResult());
    auto &initOps = init.getBody().front().getOperations();
    initOps.splice(initOps.begin(), funcOps, funcOps.begin(), itOp);

    // Construct the new `main` function.
    main.getBody().takeBody(op.body());
    Operation *yield = main.getBody().front().getTerminator();
    builder.setInsertionPoint(yield);
    builder.create<func::ReturnOp>();
    yield->erase();

    main.insertArgument(0, tupleType, /*argAttrs=*/nullptr, main.getLoc());
    builder.setInsertionPointToStart(&main.getBody().front());
    auto mainUnpackOp =
        builder.create<UnpackOp>(packedTypes, main.getArgument(0));
    replaceWithUnpacked(valuesWithCast, mainUnpackOp, main, builder);

    // Construct the `fini` function.
    builder.setInsertionPointToStart(fini.addEntryBlock());
    auto finiUnpackOp =
        builder.create<UnpackOp>(packedTypes, fini.getArgument(0));
    auto &finiOps = fini.getBody().front().getOperations();
    finiOps.splice(finiOps.end(), funcOps, cleanup->begin(),
                   std::prev(cleanup->end()));
    replaceWithUnpacked(valuesWithCast, finiUnpackOp, fini, builder);
    builder.create<func::ReturnOp>();

    // NOTE: we need to replace these at the end so that the `values` used in
    // `replaceWithUnpacked` remain valid.
    for (BlockArgument arg : func.getArguments()) {
      arg.replaceAllUsesWith(init.getArgument(arg.getArgNumber()));
    }

    func.erase();
  }

  void replaceWithUnpacked(ArrayRef<ValuesWithCast> values, UnpackOp unpackOp,
                           func::FuncOp func, OpBuilder &builder) {
    for (auto it : llvm::enumerate(values)) {
      Value value = it.value().value;
      memref::CastOp castOp = it.value().castOp;
      Value newValue = unpackOp.getResult(it.index());
      if (castOp) {
        newValue = builder.create<memref::CastOp>(
            castOp.getLoc(), castOp.dest().getType(), newValue);
      }
      value.replaceUsesWithIf(newValue, [&](OpOperand &operand) {
        return operand.getOwner()->getParentOfType<func::FuncOp>() == func;
      });
    }
  }

  void getUsedValuesDefinedOutside(Block *block, SetVector<Value> &values) {
    block->walk([&](Operation *op) {
      for (OpOperand &operand : op->getOpOperands()) {
        if (operand.get().getParentBlock() != block)
          addUsedValue(operand, values);
      }
    });
  }

  void addUsedValue(OpOperand &operand, SetVector<Value> &values) {
    if (matchPattern(operand.get(), m_Constant())) {
      OpBuilder builder(operand.getOwner());
      Operation *op = builder.clone(*operand.get().getDefiningOp());
      operand.set(op->getResult(0));
    } else {
      values.insert(operand.get());
    }
  }
};

} // namespace

std::unique_ptr<Pass> createSplitClosurePass() {
  return std::make_unique<SplitClosurePass>();
}

} // namespace pmlc::dialect::stdx
