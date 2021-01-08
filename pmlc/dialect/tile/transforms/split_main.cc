// Copyright 2019, Intel Corporation

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir;                // NOLINT
using namespace pmlc::dialect::stdx; // NOLINT

namespace pmlc::dialect::tile {

namespace {

struct SplitMainPass : public SplitMainBase<SplitMainPass> {
  void runOnOperation() final;
};

void SplitMainPass::runOnOperation() {
  // Get the modules
  ModuleOp op = getOperation();
  // Find main
  FuncOp main = op.lookupSymbol<FuncOp>("main");
  if (!main) {
    IVLOG(1, "Split-main: no main function.");
    return;
  }

  // Make function types for init, body, fini
  auto argpackType = stdx::ArgpackType::get(&getContext());
  llvm::SmallVector<Type, 8> initArgs;
  llvm::SmallVector<Type, 8> bodyArgs;
  bodyArgs.push_back(argpackType);
  for (unsigned i = 0; i < main.getNumArguments(); i++) {
    if (main.getArgAttrOfType<IntegerAttr>(i, "tile.const")) {
      initArgs.push_back(main.getArgument(i).getType());
    } else {
      bodyArgs.push_back(main.getArgument(i).getType());
    }
  }
  auto initFuncType = FunctionType::get(initArgs, {argpackType}, &getContext());
  auto bodyFuncType =
      FunctionType::get(bodyArgs, main.getType().getResults(), &getContext());
  auto finiFuncType = FunctionType::get({argpackType}, {}, &getContext());

  // Construct actual ops
  OpBuilder builder(main);
  auto initOp = builder.create<FuncOp>(main.getLoc(), "init", initFuncType);
  auto bodyOp = builder.create<FuncOp>(main.getLoc(), "main", bodyFuncType);
  auto finiOp = builder.create<FuncOp>(main.getLoc(), "fini", finiFuncType);

  // Build init function (pack inputs + return)
  builder.setInsertionPointToStart(initOp.addEntryBlock());
  auto packOp = builder.create<stdx::PackOp>(
      main.getLoc(), TypeRange(argpackType), initOp.getArguments());
  builder.create<ReturnOp>(main.getLoc(), packOp.getResult());

  // Build body function (unpack inputs, do work of original main)
  builder.setInsertionPointToStart(bodyOp.addEntryBlock());
  auto unpackOp = builder.create<stdx::UnpackOp>(main.getLoc(), initArgs,
                                                 bodyOp.getArgument(0));
  // Splice instructions across
  auto &oldBodyOps = main.front().getOperations();
  auto &newBodyOps = bodyOp.front().getOperations();
  newBodyOps.splice(newBodyOps.end(), oldBodyOps, oldBodyOps.begin(),
                    oldBodyOps.end());
  // Hook up arguments to the correct source
  unsigned curInitArg = 0;
  unsigned curBodyArg = 0;
  for (unsigned i = 0; i < main.getNumArguments(); i++) {
    if (main.getArgAttrOfType<IntegerAttr>(i, "tile.const")) {
      main.getArgument(i).replaceAllUsesWith(unpackOp.getResult(curInitArg++));
    } else {
      main.getArgument(i).replaceAllUsesWith(
          bodyOp.getArgument(1 + curBodyArg++));
    }
  }

  // Build fini function (unpack input, do nothing)
  builder.setInsertionPointToStart(finiOp.addEntryBlock());
  builder.create<stdx::UnpackOp>(main.getLoc(), initArgs,
                                 finiOp.getArgument(0));
  builder.create<ReturnOp>(main.getLoc());

  // Erase original op
  main.erase();
}

} // namespace

std::unique_ptr<Pass> createSplitMainPass() {
  return std::make_unique<SplitMainPass>();
}

} // namespace pmlc::dialect::tile
