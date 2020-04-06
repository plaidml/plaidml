// Copyright 2020 Intel Corporation

#include <vector>

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/gpu_to_vulkan/lowering.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT[build/namespaces]

// static const char *kVulkanKernelCreate = "vkrt_kernel_create";
// static const char *kVulkanKernelAddOperand = "vkrt_kernel_add_operand";
static constexpr const char *kVulkanKernelLaunch = "vkrt_kernel_launch";
static constexpr const char *kKernelBinary = "__kernel_binary";
static constexpr const char *kKernelEntryPoint = "__kernel_entry";

namespace pmlc::conversion::gpu_to_vulkan {

// void vkrt_kernel_launch(         //
//     void *binaryData,            //
//     int32_t binarySize,          //
//     const char *entryPoint,      //
//     int32_t numWorkGroupX,       //
//     int32_t numWorkGroupY,       //
//     int32_t numWorkGroupZ,       //
//     /*UnrankedMemRef[] args*/... //
// );

struct GpuLaunchFuncConversion : public OpRewritePattern<gpu::LaunchFuncOp> {
  using OpRewritePattern<gpu::LaunchFuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::LaunchFuncOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto *llvmDialect =
        op.getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    auto i32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);
    // auto ptrTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    // auto voidTy = LLVM::LLVMType::getVoidTy(llvmDialect);

    std::string data{0, 1, 2, 0, 3};
    auto blobPtr =
        LLVM::createGlobalString(loc, rewriter, kKernelBinary, data,
                                 LLVM::Linkage::Internal, llvmDialect);

    auto binarySizeAttr = rewriter.getI32IntegerAttr(data.size());
    auto binarySize =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, binarySizeAttr);

    const char *kernelName = "add_kernel";
    auto entryPoint =
        LLVM::createGlobalString(loc, rewriter, kKernelEntryPoint, kernelName,
                                 LLVM::Linkage::Internal, llvmDialect);

    // auto i8Ty = rewriter.getIntegerType(8);
    // auto i32Ty = rewriter.getIntegerType(32);

    // auto blobTy =
    //     RankedTensorType::get(static_cast<int64_t>(blobData.size()), i8Ty);
    // auto blobAttr = DenseIntElementsAttr::get(blobTy, blobData);
    // auto blobAttr = rewriter.getStringAttr(blobData);

    // auto blob = rewriter.create<ConstantOp>(loc, blobAttr);

    // auto unrankedTy = RankedTensorType::get({-1}, i8Ty);
    // auto unranked = rewriter.create<TensorCastOp>(loc, blob, unrankedTy);

    // SmallVector<Value, 1> createOperands{unranked};
    // SmallVector<Type, 1> createResults{i32Ty};
    // auto kernel = rewriter.create<CallOp>(loc, kVulkanKernelCreate,
    //                                       createResults, createOperands);
    // rewriter.replaceOp(op, kernel.getResults());

    // for (auto operand : op.operands()) {
    //   rewriter.create<CallOp>(loc, kVulkanKernelAddOperand, ArrayRef<Type>{},
    //                           operand);
    // }

    auto blockSizeX = getConstant(rewriter, loc, op.blockSizeX(), i32Ty);
    if (!blockSizeX)
      return op.emitOpError("blockSizeX is not constant");

    auto blockSizeY = getConstant(rewriter, loc, op.blockSizeY(), i32Ty);
    if (!blockSizeY)
      return op.emitOpError("blockSizeY is not constant");

    auto blockSizeZ = getConstant(rewriter, loc, op.blockSizeZ(), i32Ty);
    if (!blockSizeZ)
      return op.emitOpError("blockSizeZ is not constant");

    SmallVector<Value, 9> args{blobPtr,    binarySize, entryPoint,
                               blockSizeX, blockSizeY, blockSizeZ};

    for (auto operand : op.operands()) {
      args.push_back(getOperand(rewriter, loc, operand, llvmDialect));
    }

    auto callee = rewriter.getSymbolRefAttr(kVulkanKernelLaunch);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, ArrayRef<Type>{}, callee,
                                              args);
    return success();
  }

  Value getOperand(PatternRewriter &rewriter, Location loc, Value operand,
                   LLVM::LLVMDialect *llvmDialect) const {
    auto memRefType = operand.getType().cast<MemRefType>();
    IVLOG(1, "operand: " << mlir::debugString(memRefType));
    LLVMTypeConverter typeConverter(operand.getContext());
    // auto source = typeConverter.convertType(memRefType);
    auto unrankedTy = UnrankedTensorType::get(memRefType.getElementType());
    auto target = typeConverter.convertType(unrankedTy);
    // ptr = AllocaOp sizeof(MemRefDescriptor)
    auto ptr = typeConverter.promoteOneMemRefDescriptor(loc, operand, rewriter);
    // voidptr = BitCastOp srcType* to void*
    auto ptrTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    auto voidPtr =
        rewriter.create<LLVM::BitcastOp>(loc, ptrTy, ptr).getResult();
    // rank = ConstantOp srcRank
    auto rankVal = rewriter.create<LLVM::ConstantOp>(
        loc, typeConverter.convertType(rewriter.getIntegerType(64)),
        rewriter.getI64IntegerAttr(memRefType.getRank()));
    // undef = UndefOp
    auto memRefDesc = UnrankedMemRefDescriptor::undef(rewriter, loc, target);
    // d1 = InsertValueOp undef, rank, 0
    memRefDesc.setRank(rewriter, loc, rankVal);
    // d2 = InsertValueOp d1, voidptr, 1
    memRefDesc.setMemRefDescPtr(rewriter, loc, voidPtr);
    return memRefDesc;
  }

  Value getConstant(PatternRewriter &rewriter, Location loc, Value value,
                    Type type) const {
    IntegerAttr attr;
    auto op = value.getDefiningOp();
    if (!op)
      return Value();
    if (!m_Constant(&attr).match(op))
      return Value();
    return rewriter.create<LLVM::ConstantOp>(loc, type, attr);
  }
};

struct LowerGpuToVulkanCalls : public ModulePass<LowerGpuToVulkanCalls> {
  LowerGpuToVulkanCalls() {}

  void runOnModule() final {
    declareVulkanRuntimeCalls();

    auto &context = getContext();
    OwningRewritePatternList patterns;
    patterns.insert<GpuLaunchFuncConversion>(&context);
    applyPatternsGreedily(getModule(), patterns);
  }

  void declareVulkanRuntimeCalls() {
    auto llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    auto ptrTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    auto i32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);
    auto voidTy = LLVM::LLVMType::getVoidTy(llvmDialect);

    auto module = getModule();
    auto loc = module.getLoc();
    OpBuilder builder(module.getBodyRegion());
    // ArrayRef<Type> emptyTypes{};
    // ArrayRef<NamedAttribute> attrs{};
    // auto i8Ty = builder.getIntegerType(8);
    // auto i32Ty = builder.getIntegerType(32);

    if (!module.lookupSymbol(kVulkanKernelLaunch)) {
      SmallVector<LLVM::LLVMType, 3> params{/*binaryData=*/ptrTy,
                                            /*binarySize=*/i32Ty,
                                            /*entryPoint=*/ptrTy,
                                            /*numWorkGroupX=*/i32Ty,
                                            /*numWorkGroupY=*/i32Ty,
                                            /*numWorkGroupZ=*/i32Ty};
      auto funcTy =
          LLVM::LLVMType::getFunctionTy(voidTy, params, /*isVarArg=*/true);
      builder.create<LLVM::LLVMFuncOp>(loc, kVulkanKernelLaunch, funcTy);
    }

    // UnrankedMemRefType:
    // {getIndexType(), LLVM::LLVMType::getInt8PtrTy(llvmDialect)};

    //   if (!module.lookupSymbol(kVulkanKernelAddOperand)) {
    //     SmallVector<LLVM::LLVMType, 2> params{
    //         /*kernel=*/ptrTy,
    //         /*memref=*/getUnrankedMemRefType(llvmDialect)};
    //     auto funcTy = LLVM::LLVMType::getFunctionTy(ptrTy, params, false);
    //     builder.create<LLVM::LLVMFuncOp>(loc, kVulkanKernelAddOperand,
    //     funcTy);
    //   }

    //   if (!module.lookupSymbol(kVulkanKernelLaunch)) {
    //     SmallVector<LLVM::LLVMType, 1> params{/*kernel=*/ptrTy};
    //     auto funcTy = LLVM::LLVMType::getFunctionTy(voidTy, params, false);
    //     builder.create<LLVM::LLVMFuncOp>(loc, kVulkanKernelLaunch, funcTy);
    //   }
  }

  LLVM::LLVMType getUnrankedMemRefType(LLVM::LLVMDialect *llvmDialect) {
    auto rankTy = LLVM::LLVMType::getInt64Ty(llvmDialect);
    auto ptrTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    return LLVM::LLVMType::getStructTy(rankTy, ptrTy);
  }
};

std::unique_ptr<Pass> createLowerGpuToVulkanCalls() {
  return std::make_unique<LowerGpuToVulkanCalls>();
}

static PassRegistration<LowerGpuToVulkanCalls>
    legalize_pass("convert-gpu-to-vulkan",
                  "Convert GPU dialect host-side to Vulkan runtime calls");

} // namespace pmlc::conversion::gpu_to_vulkan
