// Copyright 2021 Intel Corporation

#include "pmlc/target/x86/pipeline.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/linalg_to_pxa/passes.h"
#include "pmlc/conversion/pxa_to_affine/passes.h"
#include "pmlc/conversion/stdx_to_llvm/passes.h"
#include "pmlc/conversion/tile_to_linalg/passes.h"
#include "pmlc/dialect/layer/transforms/passes.h"
#include "pmlc/dialect/linalgx/transforms/passes.h"
#include "pmlc/dialect/pml/transforms/passes.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/pxa/transforms/stencil.h"
#include "pmlc/dialect/stdx/transforms/passes.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/dialect/xsmm/ir/ops.h"
#include "pmlc/target/x86/heatmap.h"
#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/target/x86/passes.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

#include "omp.h" // NOLINT

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace layer = dialect::layer;
namespace linalgx = dialect::linalgx;
namespace pml = dialect::pml;
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
  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();

    LowerToLLVMOptions options(context);
    options.emitCWrappers = true;
    LLVMTypeConverter converter(context, options);

    RewritePatternSet patterns(context);
    populateExpandTanhPattern(patterns);
    // populateMathPolynomialApproximationPatterns(patterns);
    populateMathToLibmConversionPatterns(patterns, /*benefit=*/1);
    populateXSMMToLLVMConversionPatterns(converter, patterns);
    conversion::stdx_to_llvm::populateStdXToLLVMConversionPatterns(converter,
                                                                   patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    arith::populateArithmeticToLLVMConversionPatterns(converter, patterns);
    populateMathToLLVMConversionPatterns(converter, patterns);
    populateMemRefToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    populateOpenMPToLLVMConversionPatterns(converter, patterns);

    LLVMConversionTarget target(*context);
    target.addDynamicallyLegalOp<omp::ParallelOp, omp::WsLoopOp>(
        [&](Operation *op) { return converter.isLegal(&op->getRegion(0)); });
    target.addLegalOp<omp::BarrierOp, omp::FlushOp, omp::TaskyieldOp,
                      omp::TaskwaitOp, omp::TerminatorOp>();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct CollapseParallelLoopsPass
    : public CollapseParallelLoopsBase<CollapseParallelLoopsPass> {
  void runOnOperation() final {
    getOperation().walk([&](scf::ParallelOp op) {
      SmallVector<std::vector<unsigned>, 3> combinedLoops;
      std::vector<unsigned> dims;
      for (unsigned i = 0; i < op.getNumLoops(); i++)
        dims.push_back(i);
      combinedLoops.push_back(dims);
      collapseParallelLoops(op, combinedLoops);
    });
  }
};

static APFloat convertFloatUsingType(double value, FloatType type) {
  bool losesInfo = false;
  APFloat apValue(value);
  apValue.convert(type.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                  &losesInfo);
  return apValue;
}

struct FoldConstantCastPass
    : public FoldConstantCastBase<FoldConstantCastPass> {
  void runOnOperation() final {
    getOperation().walk([&](CastOpInterface op) {
      Attribute attr;
      if (matchPattern(op->getOperand(0), m_Constant(&attr))) {
        OpBuilder builder(op);
        Value result = op->getResult(0);
        Type type = result.getType();
        if (auto floatType = type.dyn_cast<FloatType>()) {
          if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
            APFloat value = convertFloatUsingType(intAttr.getInt(), floatType);
            auto constOp = builder.create<arith::ConstantFloatOp>(
                op->getLoc(), value, floatType);
            result.replaceAllUsesWith(constOp);
          }
        } else if (auto intType = type.dyn_cast<IntegerType>()) {
          if (auto floatAttr = attr.dyn_cast<FloatAttr>()) {
            int64_t value = static_cast<int64_t>(floatAttr.getValueAsDouble());
            auto constOp = builder.create<arith::ConstantIntOp>(op->getLoc(),
                                                                value, intType);
            result.replaceAllUsesWith(constOp);
          }
        }
      }
    });
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

  void runOnOperation() final {
    if (!numThreads.getValue()) {
      numThreads = std::thread::hardware_concurrency();
    }
    IVLOG(3, "StencilTppGemmPass> numThreads: " << numThreads.getValue());
    IVLOG(3, "StencilTppGemmPass> isBatched: " << isBatched.getValue());
    getOperation().walk([this](AffineParallelOp op) {
      // TODO: check LogicalResult
      (void)pxa::applyStencilGEMM(op, numThreads.getValue(),
                                  isBatched.getValue(), heatmapCostTransposed);
    });
  }
};

std::unique_ptr<Pass> createCollapseParallelLoopsPass() {
  return std::make_unique<CollapseParallelLoopsPass>();
}

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

std::unique_ptr<Pass> createFoldConstantCastPass() {
  return std::make_unique<FoldConstantCastPass>();
}

struct Options : public PassPipelineOptions<Options> {
  Option<unsigned> numThreads{*this, "threads",
                              llvm::cl::desc("Number of threads")};

  unsigned getNumThreads() const {
    return numThreads ? numThreads.getValue() : omp_get_max_threads();
  }
};

void pipelineBuilderStage1(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(layer::createInlineLayersPass());
  pm.addNestedPass<func::FuncOp>(tile::createAlgebraicOptPass());
  pm.addNestedPass<func::FuncOp>(tile::createComputeBoundsPass());
  pm.addNestedPass<func::FuncOp>(tile::createPadConstraintsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  std::string schedulePath = util::getEnvVar("PLAIDML_SCHEDULE_PATH");
  if (!schedulePath.empty()) {
    pm.addPass(pml::createLoadModulePass(/*path=*/schedulePath));
    pm.addPass(createCanonicalizerPass());
    pm.addPass(pml::createApplyRulesPass(/*module=*/"schedule"));
  }

  pm.addPass(pmlc::conversion::tile_to_linalg::createLowerTileToLinalgPass());
  pm.addNestedPass<func::FuncOp>(linalgx::createRegulateDepthwisePass());
  if (!util::getEnvVar("PLAIDML_REORDER").empty())
    pm.addNestedPass<func::FuncOp>(createReorderLayoutsPass());
  else
    pm.addNestedPass<func::FuncOp>(createReorderWeightLayoutsPass());

  pm.addPass(stdx::createMainClosurePass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  // TODO: Need to investigate why enabling this pass
  // breaks tests.
  // pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void pipelineBuilderStage2(OpPassManager &pm, const Options &options) {
  unsigned maxThreads = options.getNumThreads();
  IVLOG(1, "Number of threads: " << maxThreads);

  pm.addPass(pmlc::conversion::linalg_to_pxa::createLowerLinalgToPXAPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(layer::createInlineLayersPass());

  pm.addNestedPass<func::FuncOp>(
      createStencilTppGemmPass(/*numThreads=*/maxThreads,
                               /*isBatched=*/true));
  pm.addNestedPass<func::FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(
      pxa::createFusionPass(/*memoryActivityThreshold=*/0,
                            /*minimumThreads=*/maxThreads,
                            /*exactlyMatch=*/false,
                            /*tiledFusion=*/true,
                            /*loopDepth=*/0,
                            /*singleOutput=*/false,
                            /*avoidReductionIndexes=*/true));
  pm.addNestedPass<func::FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(
      pxa::createFusionPass(/*memoryActivityThreshold=*/0,
                            /*minimumThreads=*/maxThreads,
                            /*exactlyMatch=*/false,
                            /*tiledFusion=*/false,
                            /*loopDepth=*/1,
                            /*singleOutput=*/false,
                            /*avoidReductionIndexes=*/true));
  pm.addNestedPass<func::FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(pxa::createTileAccumulatePass());
  pm.addNestedPass<func::FuncOp>(
      pxa::createAffineNormalizePass(/*promote=*/false));
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(pxa::createCPUThreadPass(maxThreads));
  pm.addNestedPass<func::FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(pxa::createMemRefDataFlowOptPass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(pxa::createLocalizePass());
  pm.addNestedPass<func::FuncOp>(pxa::createResizeTmpsPass());
  pm.addPass(pxa::createDeallocPlacementPass());
  pm.addNestedPass<func::FuncOp>(
      pxa::createAffineNormalizePass(/*promote=*/true,
                                     /*denest=*/true));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<func::FuncOp>(createStencilSplitPass());
  pm.addNestedPass<func::FuncOp>(createStencilTppUnaryPass());
  pm.addNestedPass<func::FuncOp>(createStencilTppBinaryPass());
  pm.addNestedPass<func::FuncOp>(pxa::createAffineNormalizePass());

  if (pmlc::util::getEnvVar("PLAIDML_PROFILE") == "1")
    pm.addPass(createProfileKernelsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createPRNGLinkingPass());
  if (!pmlc::util::getEnvVar("PLAIDML_SHAPE_ANALYSIS_OUTPUT").empty())
    pm.addPass(createShapeAnalysisPass());
}

void pipelineBuilderStage3(OpPassManager &pm) {
  pm.addPass(createLowerPXAToAffinePass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerAffinePass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<func::FuncOp>(createCollapseParallelLoopsPass());
  pm.addPass(createConvertSCFToOpenMPPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<func::FuncOp>(createFoldConstantCastPass());

  pm.addPass(stdx::createSplitClosurePass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void pipelineBuilderStage4(OpPassManager &pm) {
  if (pmlc::util::getEnvVar("PLAIDML_BOUNDS_CHECK") == "1")
    pm.addNestedPass<func::FuncOp>(stdx::createBoundsCheckPass());
  pm.addPass(createLowerToLLVMPass());
  pm.addPass(createTraceLinkingPass());
  if (pmlc::util::getEnvVar("PLAIDML_PROFILE") == "1")
    pm.addPass(createProfileLinkingPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void pipelineBuilder(OpPassManager &pm) {
  Options options;
  pipelineBuilderStage1(pm);
  pipelineBuilderStage2(pm, options);
  pipelineBuilderStage3(pm);
  pipelineBuilderStage4(pm);
}

static PassPipelineRegistration<> registerStage1Pipeline("x86-stage1",
                                                         "x86 Stage1 Pipeline",
                                                         pipelineBuilderStage1);

static PassPipelineRegistration<Options>
    registerStage2Pipeline("x86-stage2", "x86 Stage2 Pipeline",
                           pipelineBuilderStage2);

static PassPipelineRegistration<> registerStage3Pipeline("x86-stage3",
                                                         "x86 Stage3 Pipeline",
                                                         pipelineBuilderStage3);

static PassPipelineRegistration<> registerStage4Pipeline("x86-stage4",
                                                         "x86 Stage4 Pipeline",
                                                         pipelineBuilderStage4);

} // namespace pmlc::target::x86
