// Copyright 2020 Intel Corporation
#include "pmlc/target/x86/pipeline.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/pxa_to_affine/passes.h"
#include "pmlc/conversion/scf_to_omp/passes.h"
#include "pmlc/conversion/stdx_to_llvm/passes.h"
#include "pmlc/conversion/tile_to_pxa/passes.h"
#include "pmlc/dialect/layer/transforms/passes.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/pxa/transforms/stencil.h"
#include "pmlc/dialect/stdx/transforms/passes.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/dialect/xsmm/ir/ops.h"
#include "pmlc/target/x86/heatmap.h"
#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/target/x86/passes.h"
#include "pmlc/transforms/passes.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

#include "omp.h" // NOLINT

#include "pmlc/target/x86/utils.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace layer = dialect::layer;
namespace pxa = dialect::pxa;
namespace stdx = dialect::stdx;
namespace tile = dialect::tile;
namespace xsmm = dialect::xsmm;

namespace {

struct LowerPXAToAffinePass
    : public ConvertPXAToAffineBase<LowerPXAToAffinePass> {
  void runOnOperation() final {
    MLIRContext &context = getContext();
    conversion::pxa_to_affine::PXAToAffineConversionTarget target(context);
    target.addLegalDialect<xsmm::XSMMDialect>();

    RewritePatternSet patterns(&context);
    populatePXAGemmToXSMMConversionPatterns(patterns);
    conversion::pxa_to_affine::populatePXAToAffineConversionPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      getOperation().dump();
      emitError(UnknownLoc::get(&context), "Error lowering pxa -> affine\n");
      signalPassFailure();
    }
  }
};

struct ConvertStandardToLLVMPass
    : public ConvertStandardToLLVMBase<ConvertStandardToLLVMPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();

    LowerToLLVMOptions options(context);
    options.emitCWrappers = true;
    LLVMTypeConverter typeConverter(context, options);

    RewritePatternSet patterns(context);
    populateExpandTanhPattern(patterns);
    populateXSMMToLLVMConversionPatterns(typeConverter, patterns);
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    conversion::stdx_to_llvm::populateStdXToLLVMConversionPatterns(
        typeConverter, patterns);
    populateOpenMPToLLVMConversionPatterns(typeConverter, patterns);

    LLVMConversionTarget target(*context);
    target.addDynamicallyLegalOp<omp::ParallelOp>([&](omp::ParallelOp op) {
      return typeConverter.isLegal(&op.getRegion());
    });
    target.addLegalOp<omp::TerminatorOp, //
                      omp::TaskyieldOp,  //
                      omp::FlushOp,      //
                      omp::BarrierOp,    //
                      omp::TaskwaitOp>();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

// NOTE: the stencil pass uses row-major ordering, the heatmap is
// specified in column-major ordering.
static pxa::StencilCost heatmapCostTransposed(ArrayRef<int64_t> tile,
                                              ArrayRef<Type> types) {
  // Only f32 is supported currently.
  if (llvm::any_of(types, [](Type type) {
        if (auto shapedType = type.dyn_cast<ShapedType>()) {
          type = shapedType.getElementType();
        }
        return !type.isF32();
      })) {
    return pxa::StencilCost{/*throughput=*/0.0, /*startupCost=*/0};
  }
  IVLOG(6, "Transposing row major [M N K] = ["
               << tile[0] << " " << tile[1] << " " << tile[2] << "] tile "
               << "to column major [N M K] = [" << tile[1] << " " << tile[0]
               << " " << tile[2] << "] tile "
               << "for lookup in column major x86 heatmap");
  return heatmapCost(ArrayRef<int64_t>{tile[1], tile[0], tile[2]});
}

struct StencilTppGemmPass : public StencilTppGemmBase<StencilTppGemmPass> {
  StencilTppGemmPass() = default;
  StencilTppGemmPass(unsigned numThreads, bool isBatched) {
    this->numThreads = numThreads;
    this->isBatched = isBatched;
  }

  void runOnFunction() final {
    if (!numThreads.getValue()) {
      numThreads = std::thread::hardware_concurrency();
    }
    IVLOG(3, "StencilTppGemmPass> numThreads: " << numThreads.getValue());
    IVLOG(3, "StencilTppGemmPass> isBatched: " << isBatched.getValue());
    getFunction().walk([this](AffineParallelOp op) {
      // TODO: check LogicalResult
      (void)pxa::applyStencilGEMM(op, numThreads.getValue(),
                                  isBatched.getValue(), heatmapCostTransposed);
    });
  }
};

std::unique_ptr<Pass> createStencilTppGemmPass(unsigned numThreads,
                                               bool isBatched) {
  return std::make_unique<StencilTppGemmPass>(numThreads, isBatched);
}

std::unique_ptr<Pass> createStencilTppGemmPass() {
  return std::make_unique<StencilTppGemmPass>();
}

std::unique_ptr<Pass> createLowerPXAToAffinePass() {
  return std::make_unique<LowerPXAToAffinePass>();
}

std::unique_ptr<Pass> createLowerToLLVMPass() {
  return std::make_unique<ConvertStandardToLLVMPass>();
}

struct Options : public PassPipelineOptions<Options> {
  Option<unsigned> numThreads{*this, "threads",
                              llvm::cl::desc("Number of threads")};

  unsigned getNumThreads() const {
    if (numThreads)
      return numThreads.getValue();

    unsigned numProcs = omp_get_num_procs();
    unsigned physCores = getPhysicalCoreNumber();
    IVLOG(3, "numProcs: " << numProcs);
    IVLOG(3, "physCores: " << physCores);
    if (physCores)
      numProcs = std::min(physCores, numProcs);
    return numProcs;
  }
};

void pipelineBuilderStage1(OpPassManager &pm, const Options &options) {
  unsigned maxThreads = options.getNumThreads();
  IVLOG(1, "Number of threads: " << maxThreads);

  pm.addNestedPass<FuncOp>(layer::createInlineLayersPass());
  pm.addNestedPass<FuncOp>(tile::createAlgebraicOptPass());
  pm.addNestedPass<FuncOp>(tile::createComputeBoundsPass());
  pm.addPass(tile::createSplitMainPass());
  pm.addPass(transforms::createHoistingPass());
  pm.addNestedPass<FuncOp>(tile::createPadConstraintsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(pmlc::conversion::tile_to_pxa::createLowerTileToPXAPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<FuncOp>(layer::createInlineLayersPass());

  pm.addNestedPass<FuncOp>(createStencilTppGemmPass(/*numThreads=*/maxThreads,
                                                    /*isBatched=*/true));
  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(pxa::createTileAccumulatePass());
  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass(/*promote=*/false));
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(pxa::createCPUThreadPass(maxThreads));
  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(pxa::createFusionPass(/*memoryActivityThreshold=*/0,
                                                 /*exactlyMatch=*/false,
                                                 /*tiledFusion=*/true));
  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(pxa::createMemRefDataFlowOptPass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(pxa::createLocalizePass());
  // pm.addNestedPass<FuncOp>(pxa::createResizeTmpsPass());
  pm.addPass(pxa::createDeallocPlacementPass());
  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass(/*promote=*/true,
                                                          /*denest=*/true));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<FuncOp>(createStencilTppUnaryPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createPRNGLinkingPass());
}

void pipelineBuilderStage2(OpPassManager &pm) {
  pm.addPass(createLowerPXAToAffinePass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(pmlc::conversion::scf_to_omp::createLowerSCFToOpenMPPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void pipelineBuilderStage3(OpPassManager &pm) {
  pm.addPass(createLowerToCFGPass());
  if (pmlc::util::getEnvVar("PLAIDML_BOUNDS_CHECK") == "1")
    pm.addNestedPass<FuncOp>(stdx::createBoundsCheckPass());

  pm.addPass(createLowerToLLVMPass());
  pm.addPass(createTraceLinkingPass());
}

void pipelineBuilder(OpPassManager &pm) {
  Options options;
  pipelineBuilderStage1(pm, options);
  pipelineBuilderStage2(pm);
  pipelineBuilderStage3(pm);
}

static PassPipelineRegistration<Options>
    registerStage1Pipeline("x86-stage1", "x86 Stage1 Pipeline",
                           pipelineBuilderStage1);

static PassPipelineRegistration<> registerStage2Pipeline("x86-stage2",
                                                         "x86 Stage2 Pipeline",
                                                         pipelineBuilderStage2);

static PassPipelineRegistration<> registerStage3Pipeline("x86-stage3",
                                                         "x86 Stage3 Pipeline",
                                                         pipelineBuilderStage3);

} // namespace pmlc::target::x86
