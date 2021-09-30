// Copyright 2021 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "pmlc/target/x86/pass_detail.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace {

static constexpr StringRef kInstrument = "plaidml_rt_instrument";

static FlatSymbolRefAttr lookupOrCreateFn(ModuleOp module, StringRef name,
                                          TypeRange argTypes,
                                          TypeRange resultTypes) {
  OpBuilder builder(module.getBodyRegion());
  if (auto fn = module.lookupSymbol<FuncOp>(name))
    return FlatSymbolRefAttr::get(fn);

  FunctionType funcType = builder.getFunctionType(argTypes, resultTypes);
  auto fn = builder.create<FuncOp>(module.getLoc(), name, funcType,
                                   builder.getStringAttr("private"));
  return FlatSymbolRefAttr::get(fn);
}

struct ProfileKernelsPass : public ProfileKernelsBase<ProfileKernelsPass> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    Type i64Type = IntegerType::get(context, 64);
    FlatSymbolRefAttr func =
        lookupOrCreateFn(module, kInstrument, {i64Type, i64Type}, TypeRange{});

    int64_t id = 0;
    module.walk<WalkOrder::PreOrder>([&](AffineParallelOp op) {
      Location loc = op->getLoc();

      OpBuilder builder(op);
      Value idValue = builder.create<ConstantIntOp>(loc, id++, 64);
      Value tagZero = builder.create<ConstantIntOp>(loc, 0, 64);
      builder.create<CallOp>(loc, TypeRange{}, func,
                             ValueRange{idValue, tagZero});

      builder.setInsertionPointAfter(op);
      Value tagOne = builder.create<ConstantIntOp>(loc, 1, 64);
      builder.create<CallOp>(loc, TypeRange{}, func,
                             ValueRange{idValue, tagOne});

      return WalkResult::skip();
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createProfileKernelsPass() {
  return std::make_unique<ProfileKernelsPass>();
}

} // namespace pmlc::target::x86
