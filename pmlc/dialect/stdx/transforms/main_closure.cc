// Copyright 2021, Intel Corporation

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/stdx/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::stdx {

namespace {

struct MainClosurePass : public MainClosureBase<MainClosurePass> {
  void runOnOperation() final;
};

void MainClosurePass::runOnOperation() {
  ModuleOp module = getOperation();
  func::FuncOp main = module.lookupSymbol<func::FuncOp>("main");
  if (!main) {
    IVLOG(1, "Split-main: 'main' function not found.");
    return;
  }

  SmallVector<BlockArgument> args;
  llvm::BitVector argIndices(main.getNumArguments());
  SmallVector<Type> argTypes;
  for (BlockArgument arg : main.getArguments()) {
    if (!main.getArgAttr(arg.getArgNumber(), "stdx.const")) {
      args.push_back(arg);
      argIndices.set(arg.getArgNumber());
      argTypes.push_back(arg.getType());
    }
  }

  Block *origBlock = &main.front();
  Operation *firstOp = &origBlock->front();
  func::ReturnOp returnOp = cast<func::ReturnOp>(origBlock->getTerminator());

  auto builder = ImplicitLocOpBuilder::atBlockBegin(main.getLoc(), origBlock);
  FunctionType funcType =
      builder.getFunctionType(argTypes, main.getFunctionType().getResults());
  auto closure = builder.create<stdx::ClosureOp>(funcType);

  Region &bodyRegion = closure.body();
  Block *body = new Block();
  bodyRegion.push_back(body);

  auto &oldBodyOps = origBlock->getOperations();
  auto &newBodyOps = body->getOperations();
  newBodyOps.splice(std::prev(newBodyOps.end()), oldBodyOps,
                    Block::iterator(firstOp), std::prev(oldBodyOps.end()));

  builder.setInsertionPointToEnd(body);
  auto yield = builder.create<stdx::YieldOp>(returnOp.operands());

  returnOp->setOperands({});

  for (BlockArgument arg : args) {
    BlockArgument newArg = body->addArgument(arg.getType(), arg.getLoc());
    arg.replaceAllUsesWith(newArg);
  }

  main.eraseArguments(argIndices);
  for (unsigned i = 0, e = main.getNumResults(); i < e; ++i) {
    main.eraseResult(0);
  }
}

} // namespace

std::unique_ptr<Pass> createMainClosurePass() {
  return std::make_unique<MainClosurePass>();
}

} // namespace pmlc::dialect::stdx
