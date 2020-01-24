// Copyright 2019, Intel Corporation

#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/pxa_to_affine/pxa_to_affine.h"
#include "pmlc/util/logging.h"

using namespace mlir;  // NOLINT[build/namespaces]
using pmlc::conversion::pxa_to_affine::createLowerPXAToAffinePass;

namespace pmlc::target::x86 {

namespace {

struct TraceLinkingPass : public OperationPass<TraceLinkingPass, LLVM::LLVMFuncOp> {
  void runOnOperation() override {
    auto op = getOperation();
    if (!op.getAttrOfType<UnitAttr>("trace")) {
      return;
    }
    IVLOG(1, "TraceLinkingPass");
    auto context = op.getContext();
    auto module = op.getParentOfType<ModuleOp>();
    auto ref = getOrInsertTrace(module);
    auto block = op.addEntryBlock();
    auto dialect = context->getRegisteredDialect<LLVM::LLVMDialect>();
    auto voidTy = LLVM::LLVMType::getVoidTy(dialect);
    OpBuilder builder(context);
    builder.setInsertionPointToStart(block);
    auto args = ArrayRef<Value>{};
    builder.create<LLVM::CallOp>(op.getLoc(), voidTy, ref, args);
    builder.create<LLVM::ReturnOp>(op.getLoc(), args);
  }

  FlatSymbolRefAttr getOrInsertTrace(ModuleOp module) {
    const char* symbol = "plaidml_rt_trace";
    auto context = module.getContext();
    if (module.lookupSymbol(symbol)) {
      return SymbolRefAttr::get(symbol, context);
    }
    OpBuilder builder(context);
    builder.setInsertionPointToStart(module.getBody());
    auto dialect = context->getRegisteredDialect<LLVM::LLVMDialect>();
    auto voidTy = LLVM::LLVMType::getVoidTy(dialect);
    auto msgTy = LLVM::LLVMType::getInt8PtrTy(dialect);
    auto funcType = LLVM::LLVMType::getFunctionTy(voidTy, {msgTy}, false);
    builder.create<LLVM::LLVMFuncOp>(module.getLoc(), symbol, funcType);
    return SymbolRefAttr::get(symbol, context);
  }

  static std::unique_ptr<Pass> create() { return std::make_unique<TraceLinkingPass>(); }
};

void addToPipeline(OpPassManager& pm) {
  // TODO: do optimizations here

  pm.addPass(createLowerPXAToAffinePass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  pm.addPass(createLowerAffinePass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  pm.addPass(createLowerToLLVMPass(true));
  // pm.addPass(TraceLinkingPass::create());
}

static PassPipelineRegistration<> passPipelineReg("target-cpu", "Target pipeline for CPU", addToPipeline);
static compiler::TargetRegistration targetReg("llvm_cpu", addToPipeline);

}  // namespace

}  // namespace pmlc::target::x86
