// Copyright 2020 Intel Corporation

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/pxa_to_affine/passes.h"
#include "pmlc/conversion/scf_to_omp/passes.h"
#include "pmlc/conversion/stdx_to_llvm/passes.h"
#include "pmlc/conversion/tile_to_pxa/passes.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/stdx/transforms/passes.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/dialect/xsmm/ir/ops.h"
#include "pmlc/target/x86/heatmap.h"
#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/target/x86/passes.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

#include "omp.h" // NOLINT

#include "pmlc/target/x86/utils.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace pxa = dialect::pxa;
namespace xsmm = dialect::xsmm;

namespace {

struct LowerPXAToAffinePass
    : public ConvertPXAToAffineBase<LowerPXAToAffinePass> {
  void runOnOperation() final {
    auto &ctx = getContext();
    conversion::pxa_to_affine::PXAToAffineConversionTarget target(ctx);
    target.addLegalDialect<xsmm::XSMMDialect>();

    OwningRewritePatternList patterns;
    populatePXAGemmToXSMMConversionPatterns(patterns, &ctx);
    populatePXAPrngToAffineConversionPatterns(patterns, &ctx);
    conversion::pxa_to_affine::populatePXAToAffineConversionPatterns(patterns,
                                                                     &ctx);

    if (failed(applyPartialConversion(getOperation(), target, patterns,
                                      nullptr))) {
      getOperation().dump();
      emitError(UnknownLoc::get(&ctx), "Error lowering pxa -> affine\n");
      signalPassFailure();
    }
  }
};

struct ExtractLoweringPattern
    : public OpConversionPattern<LLVM::ExtractValueOp> {
  using OpConversionPattern<LLVM::ExtractValueOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::ExtractValueOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Value src = operands[0];
    while (auto insertOp = mlir::dyn_cast_or_null<LLVM::InsertValueOp>(
               src.getDefiningOp())) {
      if (op.position() == insertOp.position()) {
        rewriter.replaceOp(op, insertOp.value());
        return success();
      }
      src = insertOp.container();
    }
    return failure();
  }
};

struct ConvertStandardToLLVMPass
    : public ConvertStandardToLLVMBase<ConvertStandardToLLVMPass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = module.getContext();

    LowerToLLVMOptions options = {
        /*useBarePtrCallConv=*/false,
        /*emitCWrappers=*/true,
        /*indexBitwidth=*/kDeriveIndexBitwidthFromDataLayout,
        /*useAlignedAlloc=*/false,
    };
    LLVMTypeConverter typeConverter(context, options);

    OwningRewritePatternList patterns;
    populateExpandTanhPattern(patterns, context);
    populateXSMMToLLVMConversionPatterns(typeConverter, patterns);
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    conversion::stdx_to_llvm::populateStdXToLLVMConversionPatterns(
        typeConverter, patterns);
    populateOpenMPToLLVMConversionPatterns(context, typeConverter, patterns);
    patterns.insert<ExtractLoweringPattern>(context);

    LLVMConversionTarget target(*context);
    target.addDynamicallyLegalOp<omp::ParallelOp>([&](omp::ParallelOp op) {
      return typeConverter.isLegal(&op.getRegion());
    });
    target.addLegalOp<omp::TerminatorOp, omp::TaskyieldOp, omp::FlushOp,
                      omp::BarrierOp, omp::TaskwaitOp>();
    target.addDynamicallyLegalOp<LLVM::ExtractValueOp>(
        [](LLVM::ExtractValueOp op) {
          return !mlir::dyn_cast_or_null<LLVM::InsertValueOp>(
              op.container().getDefiningOp());
        });
    if (failed(applyPartialConversion(module, target, patterns))) {
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
  return heatmapCost(ArrayRef<int64_t>{tile[1], tile[0], tile[2]});
}

std::unique_ptr<Pass> createXSMMStencilPass() {
  auto numThreads = std::thread::hardware_concurrency();
  return pxa::createStencilGEMMPass(numThreads, /*doBatch=*/true,
                                    heatmapCostTransposed);
}

std::unique_ptr<Pass> createLowerPXAToAffinePass() {
  return std::make_unique<LowerPXAToAffinePass>();
}

std::unique_ptr<Pass> createLowerToLLVMPass() {
  return std::make_unique<ConvertStandardToLLVMPass>();
}

void pipelineBuilder(OpPassManager &pm) {
  pm.addPass(pmlc::dialect::tile::createComputeBoundsPass());
  pm.addPass(pmlc::dialect::tile::createPadConstraintsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(pmlc::conversion::tile_to_pxa::createLowerTileToPXAPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(pxa::createStencilGEMMPass(/*numThreads=*/1, /*doBatch=*/true,
                                        heatmapCostTransposed));
  pm.addPass(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(pxa::createTileAccumulatePass());
  pm.addPass(pxa::createAffineNormalizePass(/*promote=*/false));
  pm.addPass(createCanonicalizerPass());

  // Use OMP thread count
  unsigned maxThreads = omp_get_max_threads();
  unsigned physCores = getPhysicalCoreNumber();
  if (0 != physCores) {
    maxThreads = std::min(physCores, maxThreads);
  }

  pm.addPass(pxa::createCPUThreadPass(maxThreads));

  pm.addPass(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(pxa::createFusionPass());
  pm.addPass(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());

  // Currently these MemRefDataFlowOptPass is disabled because it will turn
  // 0-dim tensors into actual floats, which do not correctly pass through
  // OpenMP due to calling convention issues.  TODO: Fix OpenMP upstream and
  // re-enable.
  // pm.addPass(pxa::createMemRefDataFlowOptPass());
  // pm.addPass(createCanonicalizerPass());

  pm.addPass(pxa::createLocalizePass());
  pm.addPass(pxa::createResizeTmpsPass());
  pm.addPass(pxa::createBufferPlacementPass());
  pm.addPass(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerPXAToAffinePass());

  // Unroll affine.for loops.
  pm.addPass(createLoopUnrollPass(
      /*unrollFactor=*/32,
      /*unrollUpToFactor=*/true));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(pmlc::conversion::scf_to_omp::createLowerSCFToOpenMPPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerToCFGPass());
  if (pmlc::util::getEnvVar("PLAIDML_BOUNDS_CHECK") == "1") {
    pm.addPass(pmlc::dialect::stdx::createBoundsCheckPass());
  }

  pm.addPass(createLowerToLLVMPass());
  pm.addPass(createTraceLinkingPass());
}

} // namespace pmlc::target::x86
