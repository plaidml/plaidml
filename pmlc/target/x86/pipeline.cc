// Copyright 2019, Intel Corporation

#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/dialect/pxa/passes.h"

using namespace mlir;  // NOLINT[build/namespaces]
using pmlc::dialect::pxa::createLowerPXAToAffinePass;

namespace pmlc::target::x86 {

static compiler::TargetRegistration pipeline("llvm_cpu", [](OpPassManager* pm) {
  // TODO: do optimizations here

  pm->addPass(createLowerPXAToAffinePass());
  pm->addNestedPass<FuncOp>(createCanonicalizerPass());
  pm->addNestedPass<FuncOp>(createCSEPass());

  pm->addPass(createLowerAffinePass());
  pm->addNestedPass<FuncOp>(createCanonicalizerPass());
  pm->addNestedPass<FuncOp>(createCSEPass());

  pm->addPass(createLowerToLLVMPass(true));
});

}  // namespace pmlc::target::x86
