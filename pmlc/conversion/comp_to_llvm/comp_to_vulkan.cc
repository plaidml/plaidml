// Copyright 2020, Intel Corporation
#include "pmlc/conversion/comp_to_llvm/passes.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/comp_to_llvm/pass_detail.h"
#include "pmlc/conversion/comp_to_llvm/utils.h"
#include "pmlc/dialect/comp/ir/dialect.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/tags.h"

namespace pmlc::conversion::comp_to_llvm {

namespace comp = pmlc::dialect::comp;
namespace gpu = mlir::gpu;
namespace LLVM = mlir::LLVM;
namespace spirv = mlir::spirv;

static constexpr const char *kVkInit = "vkInit";
static constexpr const char *kVkDeinit = "vkDeinit";
static constexpr const char *kVkRun = "vkRun";
static constexpr const char *kVkWait = "vkWait";
static constexpr const char *kVkScheduleFunc = "vkScheduleFunc";
static constexpr const char *kVkAlloc = "vkAlloc";
static constexpr const char *kVkDealloc = "vkDealloc";
static constexpr const char *kVkRead = "vkRead";
static constexpr const char *kVkWrite = "vkWrite";

namespace {
class ConvertCompToVulkanCall
    : public ConvertCompToVulkanCallBase<ConvertCompToVulkanCall> {
public:
  void runOnOperation();
};

void ConvertCompToVulkanCall::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  // Serialize SPIRV kernels.
  BinaryModulesMap modulesMap;
  if (mlir::failed(serializeSpirvKernels(module, modulesMap)))
    return signalPassFailure();

  // Populate conversion patterns.
  mlir::MLIRContext *context = &getContext();
  mlir::TypeConverter typeConverter, signatureConverter;
  mlir::OwningRewritePatternList patterns;
  populateCommonPatterns(context, typeConverter, signatureConverter, patterns);
  populateCompToVkPatterns(context, modulesMap, module, typeConverter,
                           patterns);
  // Set conversion target.
  mlir::ConversionTarget target(*context);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalDialect<mlir::StandardOpsDialect>();
  target.addIllegalDialect<comp::COMPDialect>();
  target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) -> bool {
    return signatureConverter.isSignatureLegal(op.getType());
  });
  if (mlir::failed(mlir::applyPartialConversion(module, target, patterns)))
    signalPassFailure();
  // Insert runtime function declarations.
  addCommonFunctionDeclarations(module);
  addVkFunctionDeclarations(module);
}

template <class Op>
struct ConvertCompToVkBasePattern : ConvertCompOpBasePattern<Op> {
  ConvertCompToVkBasePattern(mlir::TypeConverter &typeConverter,
                             mlir::MLIRContext *context)
      : ConvertCompOpBasePattern<Op>(comp::ExecEnvRuntime::Vulkan,
                                     typeConverter, context) {}
};

/// Pattern for converting operation to llvm function call,
/// performing type conversions for results.
/// It can also handle variadic arguments when configured with
/// `varArg` and `nonVarArgs` constructor parameters.
template <class Op>
struct ConvertToFuncCallPattern : ConvertCompToVkBasePattern<Op> {
  ConvertToFuncCallPattern(mlir::StringRef funcName,
                           mlir::TypeConverter &typeConverter,
                           mlir::MLIRContext *context, bool varArg = false,
                           unsigned nonVarArgs = 0)
      : ConvertCompToVkBasePattern<Op>(typeConverter, context),
        funcName(funcName), varArg(varArg), nonVarArgs(nonVarArgs) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override;

  mlir::StringRef funcName;
  bool varArg;
  unsigned nonVarArgs;
};

using ConvertToInitVulkan = ConvertToFuncCallPattern<comp::CreateExecEnv>;
using ConvertToDeinitVulkan = ConvertToFuncCallPattern<comp::DestroyExecEnv>;
using ConvertWait = ConvertToFuncCallPattern<comp::Wait>;
using ConvertDealloc = ConvertToFuncCallPattern<comp::Dealloc>;

/// Template pattern common for both comp::ScheduleRead and
/// comp::ScheduleWrite.
template <class Op>
struct ConvertScheduleReadWrite : ConvertCompToVkBasePattern<Op> {
  ConvertScheduleReadWrite(mlir::StringRef funcName,
                           mlir::TypeConverter &typeConverter,
                           mlir::MLIRContext *context)
      : ConvertCompToVkBasePattern<Op>(typeConverter, context),
        funcName(funcName) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override;

  mlir::StringRef funcName;
};

using ConvertScheduleRead = ConvertScheduleReadWrite<comp::ScheduleRead>;
using ConvertScheduleWrite = ConvertScheduleReadWrite<comp::ScheduleWrite>;

struct ConvertAlloc : ConvertCompToVkBasePattern<comp::Alloc> {
  using ConvertCompToVkBasePattern<comp::Alloc>::ConvertCompToVkBasePattern;

  mlir::LogicalResult
  matchAndRewrite(comp::Alloc op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

struct ConvertScheduleFunc : ConvertCompToVkBasePattern<comp::ScheduleFunc> {
  ConvertScheduleFunc(const BinaryModulesMap &modulesMap,
                      mlir::ModuleOp &module,
                      mlir::TypeConverter &typeConverter,
                      mlir::MLIRContext *context)
      : ConvertCompToVkBasePattern<comp::ScheduleFunc>(typeConverter, context),
        modulesMap(modulesMap), moduleOp(module) {}

  mlir::LogicalResult
  matchAndRewrite(comp::ScheduleFunc op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override;

  const BinaryModulesMap &modulesMap;
  mlir::ModuleOp moduleOp;
};

} // namespace

void populateCompToVkPatterns(mlir::MLIRContext *context,
                              const BinaryModulesMap &modulesMap,
                              mlir::ModuleOp module,
                              mlir::TypeConverter &typeConverter,
                              mlir::OwningRewritePatternList &patterns) {
  // Populate type conversion patterns.
  LLVM::LLVMType llvmInt8Ptr = LLVM::LLVMType::getInt8PtrTy(context);
  typeConverter.addConversion(
      [=](comp::ExecEnvType execEnvType) -> mlir::Optional<mlir::Type> {
        return llvmInt8Ptr;
      });
  typeConverter.addConversion(
      [=](comp::EventType eventType) -> mlir::Optional<mlir::Type> {
        return llvmInt8Ptr;
      });
  patterns.insert<ConvertToInitVulkan>(kVkInit, typeConverter, context);
  patterns.insert<ConvertToDeinitVulkan>(kVkDeinit, typeConverter, context);
  patterns.insert<ConvertScheduleFunc>(modulesMap, module, typeConverter,
                                       context);
  patterns.insert<ConvertWait>(kVkWait, typeConverter, context,
                               /*varArg=*/true, /*nonVarArgs=*/0);
  patterns.insert<ConvertAlloc>(typeConverter, context);
  patterns.insert<ConvertDealloc>(kVkDealloc, typeConverter, context);
  patterns.insert<ConvertScheduleRead>(kVkRead, typeConverter, context);
  patterns.insert<ConvertScheduleWrite>(kVkWrite, typeConverter, context);
}

void addVkFunctionDeclarations(mlir::ModuleOp &module) {
  mlir::Location loc = module.getLoc();
  mlir::OpBuilder builder(module.getBody()->getTerminator());
  mlir::MLIRContext *context = builder.getContext();
  LLVM::LLVMType llvmInt8Ptr = LLVM::LLVMType::getInt8PtrTy(context);
  LLVM::LLVMType llvmVoid = LLVM::LLVMType::getVoidTy(context);
  LLVM::LLVMType llvmInt32 = LLVM::LLVMType::getInt32Ty(context);

  builder.create<LLVM::LLVMFuncOp>(
      loc, kVkInit,
      LLVM::LLVMType::getFunctionTy(llvmInt8Ptr, {llvmInt8Ptr},
                                    /*isVarArg=*/false));

  builder.create<LLVM::LLVMFuncOp>(
      loc, kVkScheduleFunc,
      LLVM::LLVMType::getFunctionTy(llvmInt8Ptr,
                                    {llvmInt8Ptr, llvmInt32, llvmInt8Ptr,
                                     llvmInt32, llvmInt8Ptr, llvmInt32,
                                     llvmInt32, llvmInt32, llvmInt32},
                                    /*isVarArg=*/true));

  builder.create<LLVM::LLVMFuncOp>(
      loc, kVkRun,
      LLVM::LLVMType::getFunctionTy(llvmVoid, {llvmInt8Ptr},
                                    /*isVarArg=*/false));

  builder.create<LLVM::LLVMFuncOp>(
      loc, kVkDeinit,
      LLVM::LLVMType::getFunctionTy(llvmVoid, {llvmInt8Ptr},
                                    /*isVarArg=*/false));

  builder.create<LLVM::LLVMFuncOp>(
      loc, kVkWait,
      LLVM::LLVMType::getFunctionTy(llvmVoid, {llvmInt32},
                                    /*isVarArg=*/true));

  builder.create<LLVM::LLVMFuncOp>(
      loc, kVkAlloc,
      LLVM::LLVMType::getFunctionTy(llvmInt8Ptr,
                                    {llvmInt8Ptr, llvmInt32, llvmInt8Ptr},
                                    /*isVarArg=*/false));
  builder.create<LLVM::LLVMFuncOp>(
      loc, kVkWrite,
      LLVM::LLVMType::getFunctionTy(
          llvmInt8Ptr, {llvmInt8Ptr, llvmInt8Ptr, llvmInt8Ptr, llvmInt32},
          /*isVarArg=*/true));

  builder.create<LLVM::LLVMFuncOp>(
      loc, kVkRead,
      LLVM::LLVMType::getFunctionTy(
          llvmInt8Ptr, {llvmInt8Ptr, llvmInt8Ptr, llvmInt8Ptr, llvmInt32},
          /*isVarArg=*/true));

  builder.create<LLVM::LLVMFuncOp>(
      loc, kVkDealloc,
      LLVM::LLVMType::getFunctionTy(llvmVoid, {llvmInt8Ptr, llvmInt8Ptr},
                                    /*isVarArg=*/false));
}

template <class Op>
mlir::LogicalResult ConvertScheduleReadWrite<Op>::matchAndRewrite(
    Op op, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (!this->isMatchingRuntime(op)) {
    return mlir::failure();
  }

  constexpr unsigned nonVarArgs = 3;
  mlir::SmallVector<mlir::Value, nonVarArgs + 2> castOperands(
      operands.begin(), operands.begin() + nonVarArgs);

  // Convert host memref to pointer.
  mlir::Value hostPtr =
      this->materializeConversion(rewriter, op.getLoc(), operands[0]);
  if (!hostPtr) {
    return mlir::failure();
  }
  castOperands[0] = hostPtr;

  // Add event dependencies as variadic operands.
  LLVM::LLVMType llvmInt32Ty =
      LLVM::LLVMType::getInt32Ty(rewriter.getContext());
  mlir::Value eventsCnt = rewriter.create<LLVM::ConstantOp>(
      op.getLoc(), llvmInt32Ty,
      rewriter.getI32IntegerAttr(operands.size() - nonVarArgs));
  castOperands.push_back(eventsCnt);
  castOperands.insert(castOperands.end(), operands.begin() + nonVarArgs,
                      operands.end());

  mlir::Type llvmEventType = this->convertType(op.getType());
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(
      op.getOperation(), mlir::ArrayRef<mlir::Type>{llvmEventType},
      rewriter.getSymbolRefAttr(funcName), castOperands);
  return mlir::success();
}

mlir::LogicalResult
ConvertAlloc::matchAndRewrite(comp::Alloc op,
                              mlir::ArrayRef<mlir::Value> operands,
                              mlir::ConversionPatternRewriter &rewriter) const {
  if (!isMatchingRuntime(op))
    return mlir::failure();

  mlir::Location loc = op.getLoc();
  mlir::MemRefType resultType = op.getType().cast<mlir::MemRefType>();

  mlir::SmallVector<mlir::Value, 3> castOperands;
  // Operand 0 - execution environment.
  castOperands.push_back(operands[0]);
  // Operand 1 - size of allocated memory in bytes.
  auto shape = resultType.getShape();
  uint32_t numElement = 1;
  for (auto dim : shape)
    numElement *= dim;
  uint32_t elementTypeSize =
      llvm::divideCeil(resultType.getElementTypeBitWidth(), 8);
  mlir::Value bufferByteSize = rewriter.create<LLVM::ConstantOp>(
      loc, LLVM::LLVMType::getInt32Ty(rewriter.getContext()),
      rewriter.getI32IntegerAttr(numElement * elementTypeSize));
  castOperands.push_back(bufferByteSize);
  // Operand 2 - pointer to data on host or null.
  if (operands.size() > 1) {
    mlir::Value hostPtr = materializeConversion(rewriter, loc, operands[1]);
    if (!hostPtr)
      return mlir::failure();
    castOperands.push_back(hostPtr);
  } else {
    LLVM::LLVMType llvmPointerType =
        LLVM::LLVMType::getInt8PtrTy(rewriter.getContext());
    mlir::Value nullPtr = rewriter.create<LLVM::NullOp>(loc, llvmPointerType);
    castOperands.push_back(nullPtr);
  }

  mlir::Type llvmResultType = convertType(op.getType());
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(
      op.getOperation(), mlir::ArrayRef<mlir::Type>{llvmResultType},
      rewriter.getSymbolRefAttr(kVkAlloc), castOperands);
  return mlir::success();
}

template <class Op>
mlir::LogicalResult ConvertToFuncCallPattern<Op>::matchAndRewrite(
    Op op, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (!this->isMatchingRuntime(op)) {
    return mlir::failure();
  }

  mlir::SmallVector<mlir::Type, 1> convertedTypes;
  for (mlir::Type prevType : op.getOperation()->getResultTypes()) {
    convertedTypes.push_back(this->convertType(prevType));
  }

  if (!varArg) {
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op.getOperation(), convertedTypes, rewriter.getSymbolRefAttr(funcName),
        operands);
    return mlir::success();
  }

  mlir::SmallVector<mlir::Value, 1> newOperands(operands.begin(),
                                                operands.begin() + nonVarArgs);
  LLVM::LLVMType llvmInt32Ty =
      LLVM::LLVMType::getInt32Ty(rewriter.getContext());
  mlir::Value varArgsCnt = rewriter.create<LLVM::ConstantOp>(
      op.getLoc(), llvmInt32Ty,
      rewriter.getI32IntegerAttr(operands.size() - nonVarArgs));
  newOperands.push_back(varArgsCnt);
  newOperands.insert(newOperands.end(), operands.begin() + nonVarArgs,
                     operands.end());

  rewriter.replaceOpWithNewOp<LLVM::CallOp>(op.getOperation(), convertedTypes,
                                            rewriter.getSymbolRefAttr(funcName),
                                            newOperands);
  return mlir::success();
}

mlir::LogicalResult ConvertScheduleFunc::matchAndRewrite(
    comp::ScheduleFunc op, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {

  if (!isMatchingRuntime(op)) {
    return mlir::failure();
  }

  // Collect all following ScheduleFunc ops to a vector and later schedule a
  // Vulkan commandBuffer for them.
  std::vector<comp::ScheduleFunc> scheduleFuncOps;
  auto operation = op.getOperation();
  while (operation != nullptr) {
    if (!mlir::isa<comp::ScheduleFunc>(operation) &&
        !mlir::isa<comp::Wait>(operation)) {
      break;
    }
    if (mlir::isa<comp::ScheduleFunc>(operation)) {
      scheduleFuncOps.push_back(mlir::cast<comp::ScheduleFunc>(operation));
    }
    operation = operation->getNextNode();
  }

  for (auto &scheduleFuncOp : scheduleFuncOps) {
    rewriter.setInsertionPoint(scheduleFuncOp.getOperation());
    mlir::Location loc = scheduleFuncOp.getLoc();
    auto launchOp =
        mlir::cast<gpu::LaunchFuncOp>(scheduleFuncOp.body().front().front());
    std::string binaryName = launchOp.getKernelModuleName().str();
    std::string kernelName = launchOp.getKernelName().str();

    // Create kernel from serialized binary.
    if (modulesMap.count(binaryName) == 0) {
      return mlir::failure();
    }
    if (modulesMap.at(binaryName).kernelsNameMap.count(kernelName) == 0) {
      return mlir::failure();
    }

    LLVM::LLVMType llvmInt32Type =
        LLVM::LLVMType::getInt32Ty(rewriter.getContext());

    // containing all the args for kCreateVulkanLaunchKernelAction.
    std::vector<mlir::Value> createActionOperands{operands[0]};

    // Get subgroup size
    int64_t subgroupSize = 1;
    if (pmlc::hasIntegerTag(launchOp, "subgroupSize"))
      subgroupSize = pmlc::getIntegerTag(launchOp, "subgroupSize", 1);
    if (subgroupSize != 1) {
      IVLOG(2, "Subgroup size = " << subgroupSize);
    }

    mlir::Value subgroupSizeVal = rewriter.create<LLVM::ConstantOp>(
        loc, llvmInt32Type, rewriter.getI32IntegerAttr(subgroupSize));

    createActionOperands.push_back(subgroupSizeVal);

    mlir::Value binaryPtr, binaryBytes;
    getPtrToBinaryModule(rewriter, loc, modulesMap.at(binaryName), binaryPtr,
                         binaryBytes);
    createActionOperands.push_back(binaryPtr);
    createActionOperands.push_back(binaryBytes);
    mlir::Value namePtr = getPtrToGlobalString(
        rewriter, loc, modulesMap.at(binaryName).kernelsNameMap.at(kernelName));
    createActionOperands.push_back(namePtr);

    auto gSize = launchOp.getGridSizeOperandValues();
    auto x = gSize.x.getDefiningOp()->getAttrOfType<mlir::IntegerAttr>("value");
    auto y = gSize.y.getDefiningOp()->getAttrOfType<mlir::IntegerAttr>("value");
    auto z = gSize.z.getDefiningOp()->getAttrOfType<mlir::IntegerAttr>("value");
    mlir::Value gx = rewriter.create<LLVM::ConstantOp>(loc, llvmInt32Type, x);
    mlir::Value gy = rewriter.create<LLVM::ConstantOp>(loc, llvmInt32Type, y);
    mlir::Value gz = rewriter.create<LLVM::ConstantOp>(loc, llvmInt32Type, z);
    createActionOperands.push_back(gx);
    createActionOperands.push_back(gy);
    createActionOperands.push_back(gz);

    // transform mapped vulkan buffer to launch kernel.
    std::vector<mlir::Value> bufferOperands;
    for (unsigned argI = 0; argI < launchOp.getNumKernelOperands(); ++argI) {
      mlir::Value remappedArg =
          rewriter.getRemappedValue(launchOp.getKernelOperand(argI));
      bufferOperands.push_back(remappedArg);
    }

    mlir::Value bufferNum = rewriter.create<LLVM::ConstantOp>(
        loc, llvmInt32Type, rewriter.getI32IntegerAttr(bufferOperands.size()));
    createActionOperands.push_back(bufferNum);
    createActionOperands.insert(createActionOperands.end(),
                                bufferOperands.begin(), bufferOperands.end());

    mlir::Type llvmEventType = this->convertType(scheduleFuncOp.getType());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        scheduleFuncOp.getOperation(),
        mlir::ArrayRef<mlir::Type>{llvmEventType},
        rewriter.getSymbolRefAttr(kVkScheduleFunc), createActionOperands);
  }

  rewriter.create<LLVM::CallOp>(op.getLoc(), mlir::ArrayRef<mlir::Type>{},
                                rewriter.getSymbolRefAttr(kVkRun),
                                mlir::ArrayRef<mlir::Value>{operands[0]});
  return mlir::success();
}

std::unique_ptr<mlir::Pass> createConvertCompToVulkanPass() {
  return std::make_unique<ConvertCompToVulkanCall>();
}

} // namespace pmlc::conversion::comp_to_llvm
