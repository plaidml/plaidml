// Copyright 2021 Intel Corporation

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/target/x86/passes.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace stdx = dialect::stdx;

namespace {

struct LowerClosureToCoroPass
    : public ConvertClosureToCoroBase<LowerClosureToCoroPass> {
  void runOnFunction() final {
    MLIRContext &context = getContext();
    FuncOp func = getFunction();
    Region &bodyRegion = func.getBody();
    Block *origBlock = &func.front();

    stdx::ClosureOp closureOp = *bodyRegion.op_begin<stdx::ClosureOp>();
    if (!closureOp)
      return;

    auto coroHandleType = async::CoroHandleType::get(&context);
    ImplicitLocOpBuilder builder(closureOp.getLoc(), closureOp);
    auto coroIdOp =
        builder.create<async::CoroIdOp>(async::CoroIdType::get(&context));
    auto coroBeginOp =
        builder.create<async::CoroBeginOp>(coroHandleType, coroIdOp);
    auto coroSaveOp = builder.create<async::CoroSaveOp>(
        async::CoroStateType::get(&context), coroBeginOp);

    Block *resume = origBlock->splitBlock(closureOp);
    Block *cleanup = resume->splitBlock(std::next(Block::iterator(closureOp)));
    Block *suspended = cleanup->splitBlock(cleanup->getTerminator());
    Block *suspend = &bodyRegion.emplaceBlock();

    builder.setInsertionPointToEnd(origBlock);
    builder.create<BranchOp>(suspend);

    builder.setInsertionPointToEnd(resume);
    builder.create<BranchOp>(suspend);

    builder.setInsertionPointToEnd(cleanup);
    builder.create<async::CoroFreeOp>(coroIdOp, coroBeginOp);
    builder.create<BranchOp>(suspended);

    builder.setInsertionPointToStart(suspend);
    builder.create<async::CoroSuspendOp>(coroSaveOp, suspended, resume,
                                         cleanup);

    builder.setInsertionPointToStart(suspended);
    builder.create<async::CoroEndOp>(coroBeginOp);
    ReturnOp returnOp = cast<ReturnOp>(suspended->getTerminator());
    returnOp.operandsMutable().assign(coroBeginOp);

    func.insertResult(0, coroHandleType, /*resultAttrs=*/nullptr);
    for (BlockArgument arg : closureOp.getBody().getArguments()) {
      func.insertArgument(func.getNumArguments(), arg.getType(),
                          /*argAttrs=*/nullptr);
      arg.replaceAllUsesWith(func.getArgument(func.getNumArguments() - 1));
    }

    auto &fromOps = closureOp.getBody().front().getOperations();
    auto &intoOps = resume->getOperations();
    intoOps.splice(intoOps.begin(), fromOps, fromOps.begin(),
                   std::prev(fromOps.end()));
    closureOp.erase();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerClosureToCoroPass() {
  return std::make_unique<LowerClosureToCoroPass>();
}

} // namespace pmlc::target::x86
