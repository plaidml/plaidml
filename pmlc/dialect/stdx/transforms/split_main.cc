// Copyright 2021, Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/stdx/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::stdx {

namespace {

struct SplitMainPass : public SplitMainBase<SplitMainPass> {
  void runOnOperation() final;
};

void SplitMainPass::runOnOperation() {
  ModuleOp module = getOperation();
  FuncOp main = module.lookupSymbol<FuncOp>("main");
  if (!main) {
    IVLOG(1, "Split-main: 'main' function not found.");
    return;
  }

  SmallVector<BlockArgument> args;
  SmallVector<unsigned> argIndices;
  SmallVector<Type> argTypes;
  for (BlockArgument arg : main.getArguments()) {
    if (!main.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(), "stdx.const")) {
      args.push_back(arg);
      argIndices.push_back(arg.getArgNumber());
      argTypes.push_back(arg.getType());
    }
  }

  Block *origBlock = &main.front();
  Operation *firstOp = &origBlock->front();
  ReturnOp returnOp = cast<ReturnOp>(origBlock->getTerminator());

  auto builder = ImplicitLocOpBuilder::atBlockBegin(main.getLoc(), origBlock);
  FunctionType funcType =
      builder.getFunctionType(argTypes, main.getType().getResults());
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
    BlockArgument newArg = body->addArgument(arg.getType());
    arg.replaceAllUsesWith(newArg);
  }

  main.eraseArguments(argIndices);
  for (unsigned i = 0, e = main.getNumResults(); i < e; ++i) {
    main.eraseResult(0);
  }
}

} // namespace

std::unique_ptr<Pass> createSplitMainPass() {
  return std::make_unique<SplitMainPass>();
}

} // namespace pmlc::dialect::stdx
