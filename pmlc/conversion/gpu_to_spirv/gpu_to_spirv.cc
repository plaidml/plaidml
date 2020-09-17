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
    rewriter.replaceOpWithNewOp<spirv::GroupNonUniformBroadcastOp>(
        op, spirvType, spirv::Scope::Subgroup, operands[0], operands[1]);

    return success();
  }
};

struct StdxSubgroupBlockReadINTELOpConversion
    : public SPIRVOpLowering<stdx::SubgroupBlockReadINTELOp> {
  using SPIRVOpLowering<stdx::SubgroupBlockReadINTELOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(stdx::SubgroupBlockReadINTELOp blockReadOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    stdx::SubgroupBlockReadINTELOpAdaptor blockReadOperands(operands);
    auto memrefType = blockReadOp.memref().getType().cast<MemRefType>();
    auto loadPtr = spirv::getElementPtr(
        typeConverter, memrefType, blockReadOperands.memref(),
        blockReadOperands.indices(), blockReadOp.getLoc(), rewriter);
    auto ptrType = loadPtr.component_ptr().getType().cast<spirv::PointerType>();
    rewriter.replaceOpWithNewOp<spirv::SubgroupBlockReadINTELOp>(
        blockReadOp, ptrType.getPointeeType(), loadPtr.component_ptr());
    return success();
  }
};

struct StdxSubgroupBlockWriteINTELOpConversion
    : public SPIRVOpLowering<stdx::SubgroupBlockWriteINTELOp> {
  using SPIRVOpLowering<stdx::SubgroupBlockWriteINTELOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(stdx::SubgroupBlockWriteINTELOp blockReadOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    stdx::SubgroupBlockWriteINTELOpAdaptor blockWriteOperands(operands);
    auto memrefType = blockReadOp.memref().getType().cast<MemRefType>();
    auto storePtr = spirv::getElementPtr(
        typeConverter, memrefType, blockWriteOperands.memref(),
        blockWriteOperands.indices(), blockReadOp.getLoc(), rewriter);
    rewriter.replaceOpWithNewOp<spirv::SubgroupBlockWriteINTELOp>(
        blockReadOp, storePtr, blockWriteOperands.value());
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
struct StdxUnaryOpConversion : public SPIRVOpLowering<StdxOpTy> {
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
struct StdxBinaryOpConversion : public SPIRVOpLowering<StdxOpTy> {
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

    if (failed(applyFullConversion(kernelModules, *target, patterns)))
      return signalPassFailure();
  }
};
} // namespace

void populateStdxToSPIRVPatterns(MLIRContext *context,
                                 SPIRVTypeConverter &typeConverter,
                                 OwningRewritePatternList &patterns) {
  patterns.insert<StdxSubgroupBroadcastOpConversion,
                  StdxSubgroupBlockReadINTELOpConversion,
                  StdxSubgroupBlockWriteINTELOpConversion>(context,
                                                           typeConverter);
}

void populateStdxToSPIRVGLSLPatterns(MLIRContext *context,
                                     SPIRVTypeConverter &typeConverter,
                                     OwningRewritePatternList &patterns) {
  patterns.insert<StdxUnaryOpConversion<stdx::RoundOp, spirv::GLSLRoundOp>,
                  StdxUnaryOpConversion<stdx::FloorOp, spirv::GLSLFloorOp>,
                  StdxUnaryOpConversion<stdx::TanOp, spirv::GLSLTanOp>,
                  StdxUnaryOpConversion<stdx::SinHOp, spirv::GLSLSinhOp>,
                  StdxUnaryOpConversion<stdx::CosHOp, spirv::GLSLCoshOp>,
                  StdxUnaryOpConversion<stdx::ASinOp, spirv::GLSLAsinOp>,
                  StdxUnaryOpConversion<stdx::ACosOp, spirv::GLSLAcosOp>,
                  StdxUnaryOpConversion<stdx::ATanOp, spirv::GLSLAtanOp>,
                  StdxBinaryOpConversion<stdx::PowOp, spirv::GLSLPowOp>>(
      context, typeConverter);
}

std::unique_ptr<Pass> createGPUToSPIRVCustomPass() {
  return std::make_unique<GPUToSPIRVCustomPass>();
}

} // namespace pmlc::conversion::gpu_to_spirv
