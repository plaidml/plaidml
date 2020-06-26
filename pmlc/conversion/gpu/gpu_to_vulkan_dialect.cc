#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallString.h"

#include "pmlc/conversion/gpu/pass_detail.h"
#include "pmlc/dialect/vulkan/ir/ops.h"
#include "pmlc/util/logging.h"

namespace pmlc::conversion::gpu {
namespace gpu = mlir::gpu;
namespace spirv = mlir::spirv;
namespace LLVM = mlir::LLVM;
using mlir::applyPartialConversion;
using mlir::ArrayRef;
using mlir::CallOp;
using mlir::ConversionTarget;
using mlir::failure;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::OpRewritePattern;
using mlir::OwningRewritePatternList;
using mlir::PatternRewriter;
using mlir::SmallString;
using mlir::SmallVector;
using mlir::StringRef;
using mlir::success;
using mlir::Type;
using mlir::UnrankedMemRefType;
using mlir::Value;

static constexpr const char *kInitVulkan = "initVulkan";
static constexpr const char *kDeinitVulkan = "deinitVulkan";
static constexpr const char *kSubmitCommandBuffers = "submitCommandBuffers";
static constexpr const char *kCreateVulkanLaunchKernelAction =
    "createVulkanLaunchKernelAction";
static constexpr const char *kSetVulkanLaunchKernelAction =
    "setVulkanLaunchKernelAction";
static constexpr const char *kCreateVulkanMemoryTransferAction =
    "createVulkanMemoryTransferAction";
static constexpr const char *kAddVulkanLaunchActionToSchedule =
    "addVulkanLaunchActionToSchedule";
static constexpr const char *kBindBufferFloat32 = "bindBufferFloat32";
static constexpr const char *kBindBufferInt64 = "bindBufferInt64";

class BindBufferLowering : public OpRewritePattern<CallOp> {
public:
  using OpRewritePattern<CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CallOp op,
                                PatternRewriter &rewrite) const override {
    rewrite.replaceOpWithNewOp<pmlc::dialect::vulkan::Alloc>(
        op, rewrite.getType<pmlc::dialect::vulkan::BufferType>());
    return success();
  }
};

void populateVulkanDialectPatterns(OwningRewritePatternList &patterns,
                                   MLIRContext *ctx) {
  patterns.insert<BindBufferLowering>(ctx);
}

/// A pass to convert gpu launch op to vulkan launch call op, by creating a
/// SPIR-V binary shader from `spirv::ModuleOp` using `spirv::serialize`
/// function and attaching binary data and entry point name as an attributes to
/// created vulkan launch call op.
class ConvertGpuLaunchFuncToVulkanDialect
    : public ConvertGpuLaunchFuncToVulkanDialectBase<
          ConvertGpuLaunchFuncToVulkanDialect> {
public:
  void runOnOperation();

private:
  /// Creates a SPIR-V binary shader from the given `module` using
  /// `spirv::serialize` function.
  LogicalResult createBinaryShader(ModuleOp module,
                                   std::vector<char> &binaryShader);

  /// Creates a LLVM global for the given `name`.
  Value createEntryPointNameConstant(StringRef name, uint64_t lauchFuncIndex,
                                     Location loc, OpBuilder &builder);

  /// bind gpu.launchOp buffers to Vulkan runtime.
  LogicalResult bindBuffers(Location loc, OpBuilder &builder,
                            gpu::LaunchFuncOp launchOp);

  /// Check and transfer VkBuffers when necessary.
  LogicalResult transferBuffers(Location loc, OpBuilder &builder,
                                gpu::LaunchFuncOp launchOp);

  /// Print a single buffer.
  LogicalResult printBuffer(Location loc, OpBuilder &builder, Value &buffer);

  /// Converts the given `luanchOp` to vulkan launch call.
  void convertGpuLaunchFunc(gpu::LaunchFuncOp launchOp);

  /// Declares all needed runtime functions.
  void declareVulkanFunctions(Location loc);

  void getCachedTypes() {
    llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    llvmVoidType = LLVM::LLVMType::getVoidTy(llvmDialect);
    llvmPointerType = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    llvmInt32Type = LLVM::LLVMType::getInt32Ty(llvmDialect);
    llvmInt64Type = LLVM::LLVMType::getInt64Ty(llvmDialect);

    OpBuilder builder(getOperation());
    mlirIndexType = builder.getIndexType();
    mlirInt32Type = builder.getIntegerType(32);
    mlirInt64Type = builder.getIntegerType(64);
    mlirFloat32Type = builder.getF32Type();
  }

  mlir::Type getUnrankedMemRefType(Type &elementType) {
    return UnrankedMemRefType::get(elementType, /*memorySpace=*/0);
  }

  const char *getBufferBindingFunc(Type &elementType) {
    if (elementType.isInteger(64)) {
      return kBindBufferInt64;
    }
    if (elementType.isF32()) {
      return kBindBufferFloat32;
    }
    return nullptr;
  }

  LLVM::LLVMDialect *getLLVMDialect() { return llvmDialect; }
  LLVM::LLVMType &getLLVMVoidType() { return llvmVoidType; }
  LLVM::LLVMType &getLLVMPointerType() { return llvmPointerType; }
  LLVM::LLVMType &getLLVMInt32Type() { return llvmInt32Type; }
  LLVM::LLVMType &getLLVMInt64Type() { return llvmInt64Type; }

  mlir::Type &getMLIRFloat32Type() { return mlirFloat32Type; }
  mlir::Type &getMLIRIndexType() { return mlirIndexType; }
  mlir::Type &getMLIRInt32Type() { return mlirInt32Type; }
  mlir::Type &getMLIRInt64Type() { return mlirInt64Type; }

  LLVM::LLVMDialect *llvmDialect;
  LLVM::LLVMType llvmVoidType;
  LLVM::LLVMType llvmPointerType;
  LLVM::LLVMType llvmInt32Type;
  LLVM::LLVMType llvmInt64Type;

  mlir::Type mlirInt32Type;
  mlir::Type mlirInt64Type;
  mlir::Type mlirFloat32Type;
  mlir::Type mlirIndexType;

  uint64_t numKernel = 0;
  uint64_t lauchFuncIndex = 0;
  llvm::DenseMap<Value, llvm::SmallVector<uint64_t, 2>> bufferMap;
  mlir::Value vulkanRuntime;
};

void ConvertGpuLaunchFuncToVulkanDialect::runOnOperation() {
  getCachedTypes();
  getOperation().walk([this](gpu::LaunchFuncOp op) { numKernel++; });
  getOperation().walk(
      [this](gpu::LaunchFuncOp op) { convertGpuLaunchFunc(op); });

  // Erase `gpu::GPUModuleOp` and `spirv::Module` operations.
  for (auto gpuModule :
       llvm::make_early_inc_range(getOperation().getOps<gpu::GPUModuleOp>()))
    gpuModule.erase();

  for (auto spirvModule :
       llvm::make_early_inc_range(getOperation().getOps<spirv::ModuleOp>()))
    spirvModule.erase();

  /*
  // Convert module to vulkan dialect ops
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  OwningRewritePatternList patterns;
  populateVulkanDialectPatterns(patterns, context);
  ConversionTarget target(*context);
  target.addLegalDialect<mlir::StandardOpsDialect, LLVM::LLVMDialect,
                         gpu::GPUDialect, pmlc::dialect::vulkan::VkDialect>();
  target.addDynamicallyLegalOp<mlir::CallOp>([](mlir::CallOp op) {
    if (op.callee().equals(kBindBufferFloat32))
      return false;
    else
      return true;
  });
  if (failed(applyPartialConversion(module, target, patterns))) {
    signalPassFailure();
  }*/
}

LogicalResult ConvertGpuLaunchFuncToVulkanDialect::createBinaryShader(
    ModuleOp module, std::vector<char> &binaryShader) {
  SmallVector<uint32_t, 0> binary;
  uint64_t shader_index = 0;
  for (auto spirvModule : module.getOps<spirv::ModuleOp>()) {
    if (shader_index == lauchFuncIndex) {
      if (failed(spirv::serialize(spirvModule, binary))) {
        return failure();
      }
    }
    shader_index++;
  }
  binaryShader.resize(binary.size() * sizeof(uint32_t));
  std::memcpy(binaryShader.data(), reinterpret_cast<char *>(binary.data()),
              binaryShader.size());
  return success();
}

Value ConvertGpuLaunchFuncToVulkanDialect::createEntryPointNameConstant(
    StringRef name, uint64_t lauchFuncIndex, Location loc, OpBuilder &builder) {
  SmallString<16> shaderName(name.begin(), name.end());
  // Append `\0` to follow C style string given that
  // LLVM::createGlobalString() won't handle this directly for us.
  shaderName.push_back('\0');

  std::string entryPointGlobalName =
      (name + "_spv_entry_point_name" + std::to_string(lauchFuncIndex)).str();
  return LLVM::createGlobalString(loc, builder, entryPointGlobalName,
                                  shaderName, LLVM::Linkage::Internal,
                                  getLLVMDialect());
}

LogicalResult ConvertGpuLaunchFuncToVulkanDialect::bindBuffers(
    Location loc, OpBuilder &builder, gpu::LaunchFuncOp launchOp) {
  auto buffers = launchOp.operands();
  for (uint32_t bindIndex = 0; bindIndex < buffers.size(); bindIndex++) {
    auto buffer = buffers[bindIndex];
    if (auto memRefType = buffer.getType().dyn_cast_or_null<MemRefType>()) {
      builder.create<pmlc::dialect::vulkan::Alloc>(
          loc, builder.getType<pmlc::dialect::vulkan::BufferType>());
    } else {
      return failure();
    }
  }
  return success();
}

LogicalResult ConvertGpuLaunchFuncToVulkanDialect::transferBuffers(
    Location loc, OpBuilder &builder, gpu::LaunchFuncOp launchOp) {
  auto buffers = launchOp.operands();
  for (size_t i = 0; i < buffers.size(); i++) {
    for (auto pair : bufferMap) {
      if (pair.first == buffers[i]) {
        Value dst_index = builder.create<LLVM::ConstantOp>(
            loc, getLLVMInt64Type(), builder.getI64IntegerAttr(lauchFuncIndex));
        Value dst_binding = builder.create<LLVM::ConstantOp>(
            loc, getLLVMInt64Type(), builder.getI64IntegerAttr(i));
        Value src_index = builder.create<LLVM::ConstantOp>(
            loc, getLLVMInt64Type(), builder.getI64IntegerAttr(pair.second[0]));
        Value src_binding = builder.create<LLVM::ConstantOp>(
            loc, getLLVMInt64Type(), builder.getI64IntegerAttr(pair.second[1]));

        builder.create<LLVM::CallOp>(
            loc, ArrayRef<Type>{},
            builder.getSymbolRefAttr(kCreateVulkanMemoryTransferAction),
            ArrayRef<Value>{vulkanRuntime, src_index, src_binding, dst_index,
                            dst_binding});
      }
    }
    llvm::SmallVector<uint64_t, 2> second;
    second.append({lauchFuncIndex, i});
    bufferMap[buffers[i]] = second;
  }
  return success();
}

void ConvertGpuLaunchFuncToVulkanDialect::declareVulkanFunctions(Location loc) {
  ModuleOp module = getOperation();
  OpBuilder builder(module.getBody()->getTerminator());

  if (!module.lookupSymbol(kInitVulkan)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kInitVulkan,
        LLVM::LLVMType::getFunctionTy(getLLVMPointerType(), {},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kDeinitVulkan)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kDeinitVulkan,
        LLVM::LLVMType::getFunctionTy(getLLVMVoidType(), {getLLVMPointerType()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kAddVulkanLaunchActionToSchedule)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kAddVulkanLaunchActionToSchedule,
        LLVM::LLVMType::getFunctionTy(getLLVMVoidType(), {getLLVMPointerType()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kSubmitCommandBuffers)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSubmitCommandBuffers,
        LLVM::LLVMType::getFunctionTy(getLLVMVoidType(), {getLLVMPointerType()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kCreateVulkanMemoryTransferAction)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kCreateVulkanMemoryTransferAction,
        LLVM::LLVMType::getFunctionTy(getLLVMVoidType(),
                                      {getLLVMPointerType(), getLLVMInt64Type(),
                                       getLLVMInt64Type(), getLLVMInt64Type(),
                                       getLLVMInt64Type()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kCreateVulkanLaunchKernelAction)) {
    auto &ctx = getContext();
    builder.create<FuncOp>(
        loc, kCreateVulkanLaunchKernelAction,
        FunctionType::get(
            {ArrayRef<Type>{getLLVMPointerType(), getLLVMPointerType(),
                            getLLVMInt32Type(), getLLVMPointerType(),
                            getMLIRIndexType(), getMLIRIndexType(),
                            getMLIRIndexType()}},
            {}, &ctx),
        ArrayRef<std::pair<mlir::Identifier, mlir::Attribute>>());
  }

  if (!module.lookupSymbol(kSetVulkanLaunchKernelAction)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetVulkanLaunchKernelAction,
        LLVM::LLVMType::getFunctionTy(getLLVMVoidType(), {getLLVMPointerType()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kBindBufferFloat32)) {
    auto &ctx = getContext();
    builder.create<FuncOp>(
        loc, kBindBufferFloat32,
        FunctionType::get(
            {ArrayRef<Type>{getLLVMPointerType(), getLLVMInt32Type(),
                            getLLVMInt32Type(),
                            getUnrankedMemRefType(getMLIRFloat32Type())}},
            {}, &ctx),
        ArrayRef<std::pair<mlir::Identifier, mlir::Attribute>>());
  }

  if (!module.lookupSymbol(kBindBufferInt64)) {
    auto &ctx = getContext();
    builder.create<FuncOp>(
        loc, kBindBufferInt64,
        FunctionType::get(
            {ArrayRef<Type>{getLLVMPointerType(), getLLVMInt32Type(),
                            getLLVMInt32Type(),
                            getUnrankedMemRefType(getMLIRInt64Type())}},
            {}, &ctx),
        ArrayRef<std::pair<mlir::Identifier, mlir::Attribute>>());
  }
}

void ConvertGpuLaunchFuncToVulkanDialect::convertGpuLaunchFunc(
    gpu::LaunchFuncOp launchOp) {
  ModuleOp module = getOperation();
  OpBuilder builder(launchOp);
  Location loc = launchOp.getLoc();

  if (lauchFuncIndex == 0) {
    auto initVulkanCall = builder.create<pmlc::dialect::vulkan::InitVulkanCall>(
        loc, builder.getType<pmlc::dialect::vulkan::BufferType>(),
        builder.getSymbolRefAttr(kInitVulkan));
    vulkanRuntime = initVulkanCall.getResult();
  }

  // Serialize `spirv::Module` into binary form.
  std::vector<char> binary;
  if (failed(createBinaryShader(module, binary)))
    return signalPassFailure();

  auto shaderModule =
      builder.create<pmlc::dialect::vulkan::CreateShaderModuleOp>(
          loc, pmlc::dialect::vulkan::ShaderModuleType::get(loc->getContext()),
          mlir::StringAttr::get({binary.data(), binary.size()},
                                loc->getContext()));

  // Create LLVM constant for the size of SPIR-V binary shader.
  Value binarySize = builder.create<LLVM::ConstantOp>(
      loc, getLLVMInt32Type(), builder.getI32IntegerAttr(binary.size()));

  // Create LLVM global with entry point name.
  Value entryPointName = createEntryPointNameConstant(
      launchOp.getKernelName(), lauchFuncIndex, loc, builder);

  auto gridSize = launchOp.getGridSizeOperandValues();
  // Create createVulkanLaunchKernelAction.
  builder.create<pmlc::dialect::vulkan::CreateVulkanLaunchKernelAction>(
      loc, builder.getSymbolRefAttr(kCreateVulkanLaunchKernelAction),
      ArrayRef<Value>{vulkanRuntime, shaderModule, binarySize, entryPointName,
                      gridSize.x, gridSize.y, gridSize.z});

  /// bind gpu.launchOp buffers to Vulkan runtime.
  if (failed(bindBuffers(loc, builder, launchOp))) {
    return signalPassFailure();
  }

  builder.create<pmlc::dialect::vulkan::SetLaunchKernelAction>(
      loc, builder.getSymbolRefAttr(kSetVulkanLaunchKernelAction),
      ArrayRef<Value>{vulkanRuntime});

  // Check and transfer VkBuffers when necessary.
  if (failed(transferBuffers(loc, builder, launchOp))) {
    return signalPassFailure();
  }

  builder.create<pmlc::dialect::vulkan::AddVulkanLaunchActionToSchedule>(
      loc, builder.getSymbolRefAttr(kAddVulkanLaunchActionToSchedule),
      ArrayRef<Value>{vulkanRuntime});

  if (lauchFuncIndex == numKernel - 1) {
    builder.create<pmlc::dialect::vulkan::SubmitCommandBuffers>(
        loc, builder.getSymbolRefAttr(kSubmitCommandBuffers),
        ArrayRef<Value>{vulkanRuntime});

    builder.create<pmlc::dialect::vulkan::DeinitVulkan>(
        loc, builder.getSymbolRefAttr(kDeinitVulkan),
        ArrayRef<Value>{vulkanRuntime});
  }

  launchOp.erase();
  lauchFuncIndex++;
}

std::unique_ptr<mlir::Pass> createConvertGpuLaunchFuncToVulkanDialectPass() {
  return std::make_unique<ConvertGpuLaunchFuncToVulkanDialect>();
}

} // namespace pmlc::conversion::gpu
