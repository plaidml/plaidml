// Copyright 2020, Intel Corporation

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/target/x86/pass_detail.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace pxa = dialect::pxa;

namespace {

// TODO: Lorenzo fix style use camelCase.
struct PRNGLinkingPass : public PRNGLinkingBase<PRNGLinkingPass> {
  void runOnOperation() override {
    getOperation().walk([](pxa::PrngOp op) {
      ModuleOp module = op->getParentOfType<ModuleOp>();
      MLIRContext *context = module.getContext();
      Location loc = op.getLoc();
      OpBuilder builder(op);

      auto resultType =
          UnrankedMemRefType::get(builder.getF32Type(), /*memorySpace=*/0);
      auto stateType = UnrankedMemRefType::get(builder.getIntegerType(32),
                                               /*memorySpace=*/0);
      auto symbol = getOrInsertFunc(builder, module, builder.getF32Type(),
                                    resultType, stateType);

      auto resultCast =
          builder.create<memref::CastOp>(loc, resultType, op.tensor());
      auto stateCast =
          builder.create<memref::CastOp>(loc, stateType, op.state());
      auto newStateCast =
          builder.create<memref::CastOp>(loc, stateType, op.new_state());

      builder.create<CallOp>(
          loc, symbol, ArrayRef<Type>{},
          ArrayRef<Value>{stateCast, resultCast, newStateCast});

      op.result_tensor().replaceAllUsesWith(op.tensor());
      op.result_state().replaceAllUsesWith(op.new_state());
      op.erase();
    });
  }

private:
  static FlatSymbolRefAttr getOrInsertFunc(OpBuilder &builder, ModuleOp module,
                                           Type elementType,
                                           UnrankedMemRefType resultType,
                                           UnrankedMemRefType stateType) {
    const char *symbol = "plaidml_rt_prng";
    MLIRContext *context = module.getContext();
    if (module.lookupSymbol(symbol)) {
      return SymbolRefAttr::get(context, symbol);
    }
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto funcType = builder.getFunctionType(
        ArrayRef<Type>{stateType, resultType, stateType}, ArrayRef<Type>{});
    builder
        .create<FuncOp>(builder.getUnknownLoc(), symbol, funcType,
                        ArrayRef<NamedAttribute>{})
        .setPrivate();
    return SymbolRefAttr::get(context, symbol);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createPRNGLinkingPass() {
  return std::make_unique<PRNGLinkingPass>();
}

} // namespace pmlc::target::x86
