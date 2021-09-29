// Copyright 2021 Intel Corporation

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "pmlc/target/x86/pass_detail.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace {

static constexpr StringRef kInstrument = "plaidml_rt_instrument";

struct ProfileKernelsPass : public ProfileKernelsBase<ProfileKernelsPass> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    Type i64Type = IntegerType::get(context, 64);
    LLVM::LLVMFuncOp func =
        LLVM::lookupOrCreateFn(module, kInstrument, {i64Type, i64Type},
                               LLVM::LLVMVoidType::get(context));

    int64_t id = 0;
    module.walk([&](omp::ParallelOp op) {
      Location loc = op->getLoc();
      OpBuilder builder(op);
      Value idValue = builder.create<LLVM::ConstantOp>(
          loc, builder.getI64Type(), builder.getIndexAttr(id++));

      Value tagZero = builder.create<LLVM::ConstantOp>(
          loc, builder.getI64Type(), builder.getIndexAttr(0));
      LLVM::createLLVMCall(builder, loc, func, ValueRange{idValue, tagZero});

      builder.setInsertionPointAfter(op);
      Value tagOne = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                      builder.getIndexAttr(1));
      LLVM::createLLVMCall(builder, loc, func, ValueRange{idValue, tagOne});
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createProfileKernelsPass() {
  return std::make_unique<ProfileKernelsPass>();
}

} // namespace pmlc::target::x86
