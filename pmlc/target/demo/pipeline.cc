// Copyright 2019, Intel Corporation

#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/pxa_to_affine/pxa_to_affine.h"
#include "pmlc/conversion/tile_to_pxa/tile_to_pxa.h"
#include "pmlc/dialect/tile/transforms/passes.h"

using namespace mlir; // NOLINT[build/namespaces]
using pmlc::conversion::pxa_to_affine::createLowerPXAToAffinePass;

namespace pmlc::target::demo {

namespace {

void addToPipeline(OpPassManager &pm) {
  pm.addPass(pmlc::dialect::tile::createComputeBoundsPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  pm.addPass(pmlc::conversion::tile_to_pxa::createLowerTileToPXAPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  // TODO: do optimizations here

  pm.addPass(createLowerPXAToAffinePass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  pm.addPass(createLowerAffinePass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
}

static PassPipelineRegistration<>
    passPipelineReg("target_demo", "Target pipeline for demonstrations",
                    addToPipeline);
static compiler::TargetRegistration targetReg("demo", addToPipeline);

} // namespace

} // namespace pmlc::target::demo
