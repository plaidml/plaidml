// Copyright 2020 Intel Corporation

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "pmlc/conversion/gpu_to_spirv/pass_detail.h"
#include "pmlc/conversion/gpu_to_spirv/passes.h"
#include "pmlc/dialect/stdx/ir/ops.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::conversion::gpu_to_spirv {
namespace stdx = dialect::stdx;

namespace {
/// Pass to lower to SPIRV that includes GPU, SCF, Std and Stdx dialects
struct StdxSubgroupBroadcastOpConversion final
    : public SPIRVOpLowering<stdx::SubgroupBroadcastOp> {
  using SPIRVOpLowering<stdx::SubgroupBroadcastOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(stdx::SubgroupBroadcastOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto stdxType = op.getResult().getType();
    auto spirvType = typeConverter.convertType(stdxType);
    rewriter.replaceOpWithNewOp<spirv::GroupBroadcastOp>(
        op, spirvType, spirv::Scope::Subgroup, operands[0], operands[1]);

    return success();
  }
};

// Convert all allocations within SPIRV code to function local allocations.
// TODO: Allocations outside of threads but inside blocks should probably be
// shared memory, but currently we never generate such allocs.
struct AllocOpPattern final : public SPIRVOpLowering<AllocOp> {
public:
  using SPIRVOpLowering<AllocOp>::SPIRVOpLowering;

  AllocOpPattern(MLIRContext *context, SPIRVTypeConverter &typeConverter)
      : SPIRVOpLowering<AllocOp>(context, typeConverter, /*benefit=*/1000) {}

  LogicalResult
  matchAndRewrite(AllocOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType allocType = operation.getType();
    unsigned memSpace = typeConverter.getMemorySpaceForStorageClass(
        spirv::StorageClass::Function);
    MemRefType allocType2 =
        MemRefType::Builder(allocType).setMemorySpace(memSpace);
    Type spirvType = typeConverter.convertType(allocType2);
    rewriter.replaceOpWithNewOp<spirv::VariableOp>(
        operation, spirvType, spirv::StorageClass::Function, nullptr);
    return success();
  }
};

template <typename StdxOpTy, typename SpirvOpTy>
struct UnaryOpConversion : public SPIRVOpLowering<StdxOpTy> {
  using SPIRVOpLowering<StdxOpTy>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(StdxOpTy op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 1);
    auto dstType = op.getResult().getType();
    rewriter.replaceOpWithNewOp<SpirvOpTy>(op, dstType, operands.front());
    return success();
  }
};

template <typename StdxOpTy, typename SpirvOpTy>
struct BinaryOpConversion : public SPIRVOpLowering<StdxOpTy> {
  using SPIRVOpLowering<StdxOpTy>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(StdxOpTy op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 2);
    auto dstType = op.getResult().getType();
    rewriter.replaceOpWithNewOp<SpirvOpTy>(op, dstType, operands.front(),
                                           operands.back());
    return success();
  }
};

// ============================================================================
// GLSL SPIRV -> Standard SPIRV
// ============================================================================
template <typename GLSLOpTy, typename NegOpTy, typename LessThanOpTy>
struct GLSLAbsOpPattern final : public SPIRVOpLowering<GLSLOpTy> {
  using SPIRVOpLowering<GLSLOpTy>::SPIRVOpLowering;

  mlir::LogicalResult
  matchAndRewrite(GLSLOpTy op, mlir::ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    mlir::Value arg = operands[0];
    mlir::Type targetType = arg.getType();
    mlir::Value cst0 = rewriter.create<spirv::ConstantOp>(
        op.getLoc(), targetType, rewriter.getIntegerAttr(targetType, 0));
    mlir::Value negArg = rewriter.create<NegOpTy>(op.getLoc(), arg);
    mlir::Value isNeg = rewriter.create<LessThanOpTy>(op.getLoc(), arg, cst0);
    rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, isNeg, negArg, arg);
    return mlir::success();
  }
};

using GLSLFAbsOpPattern = GLSLAbsOpPattern<spirv::GLSLFAbsOp, spirv::FNegateOp,
                                           spirv::FOrdLessThanOp>;
using GLSLSAbsOpPattern =
    GLSLAbsOpPattern<spirv::GLSLSAbsOp, spirv::SNegateOp, spirv::SLessThanOp>;

struct GPUToSPIRVCustomPass
    : public GPUToSPIRVCustomBase<GPUToSPIRVCustomPass> {
  void runOnOperation() final {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    SmallVector<Operation *, 1> kernelModules;
    OpBuilder builder(context);
    module.walk([&builder, &kernelModules](gpu::GPUModuleOp moduleOp) {
      // For each kernel module (should be only 1 for now, but that is not a
      // requirement here), clone the module for conversion because the
      // gpu.launch function still needs the kernel module.
      builder.setInsertionPoint(moduleOp.getOperation());
      kernelModules.push_back(builder.clone(*moduleOp.getOperation()));
    });

    auto targetAttr = spirv::lookupTargetEnvOrDefault(module);
    std::unique_ptr<ConversionTarget> target =
        spirv::SPIRVConversionTarget::get(targetAttr);

    SPIRVTypeConverter typeConverter(targetAttr);
    ScfToSPIRVContext scfContext;
    OwningRewritePatternList patterns;
    populateGPUToSPIRVPatterns(context, typeConverter, patterns);
    populateSCFToSPIRVPatterns(context, typeConverter, scfContext, patterns);
    populateStandardToSPIRVPatterns(context, typeConverter, patterns);
    populateStdxToSPIRVPatterns(context, typeConverter, patterns);
    patterns.insert<AllocOpPattern>(context, typeConverter);
    if (spirv::getMemoryModel(targetAttr) == spirv::MemoryModel::GLSL450)
      populateStdxToSPIRVGLSLPatterns(context, typeConverter, patterns);
    if (spirv::getMemoryModel(targetAttr) != spirv::MemoryModel::GLSL450) {
      populateCustomGLSLToStdSpirvPatterns(context, typeConverter, patterns);
      target->addIllegalOp<spirv::GLSLFAbsOp, spirv::GLSLSAbsOp,
                           spirv::GLSLExpOp>();
    }
    if (spirv::getMemoryModel(targetAttr) == spirv::MemoryModel::OpenCL)
      populateCustomStdToOCLSpirvPatterns(context, typeConverter, patterns);

    if (failed(applyFullConversion(kernelModules, *target, patterns)))
      return signalPassFailure();
  }
};
} // namespace

void populateStdxToSPIRVPatterns(MLIRContext *context,
                                 SPIRVTypeConverter &typeConverter,
                                 OwningRewritePatternList &patterns) {
  patterns.insert<StdxSubgroupBroadcastOpConversion>(context, typeConverter);
}

void populateStdxToSPIRVGLSLPatterns(MLIRContext *context,
                                     SPIRVTypeConverter &typeConverter,
                                     OwningRewritePatternList &patterns) {
  patterns.insert<UnaryOpConversion<stdx::RoundOp, spirv::GLSLRoundOp>,
                  UnaryOpConversion<stdx::FloorOp, spirv::GLSLFloorOp>,
                  UnaryOpConversion<stdx::TanOp, spirv::GLSLTanOp>,
                  UnaryOpConversion<stdx::SinHOp, spirv::GLSLSinhOp>,
                  UnaryOpConversion<stdx::CosHOp, spirv::GLSLCoshOp>,
                  UnaryOpConversion<stdx::ASinOp, spirv::GLSLAsinOp>,
                  UnaryOpConversion<stdx::ACosOp, spirv::GLSLAcosOp>,
                  UnaryOpConversion<stdx::ATanOp, spirv::GLSLAtanOp>,
                  BinaryOpConversion<stdx::PowOp, spirv::GLSLPowOp>>(
      context, typeConverter);
}

void populateCustomGLSLToStdSpirvPatterns(MLIRContext *context,
                                          SPIRVTypeConverter &typeConverter,
                                          OwningRewritePatternList &patterns) {
  patterns.insert<GLSLFAbsOpPattern, GLSLSAbsOpPattern>(context, typeConverter);
}

void populateCustomStdToOCLSpirvPatterns(MLIRContext *context,
                                         SPIRVTypeConverter &typeConverter,
                                         OwningRewritePatternList &patterns) {
  patterns.insert<UnaryOpConversion<mlir::ExpOp, spirv::OCLExpOp>>(
      context, typeConverter);
}

std::unique_ptr<Pass> createGPUToSPIRVCustomPass() {
  return std::make_unique<GPUToSPIRVCustomPass>();
}

} // namespace pmlc::conversion::gpu_to_spirv
