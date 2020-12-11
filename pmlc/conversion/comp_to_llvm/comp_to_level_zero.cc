// Copyright 2020, Intel Corporation
#include "pmlc/conversion/comp_to_llvm/passes.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/comp_to_llvm/pass_detail.h"
#include "pmlc/conversion/comp_to_llvm/utils.h"
#include "pmlc/dialect/comp/ir/dialect.h"

namespace pmlc::conversion::comp_to_llvm {

namespace comp = pmlc::dialect::comp;
namespace gpu = mlir::gpu;
namespace LLVM = mlir::LLVM;
namespace spirv = mlir::spirv;

static constexpr const char *kLevelZeroCreate = "levelZeroCreate";
static constexpr const char *kLevelZeroDestroy = "levelZeroDestroy";
static constexpr const char *kLevelZeroAlloc = "levelZeroAlloc";
static constexpr const char *kLevelZeroDealloc = "levelZeroDealloc";
static constexpr const char *kLevelZeroRead = "levelZeroRead";
static constexpr const char *kLevelZeroWrite = "levelZeroWrite";
static constexpr const char *kLevelZeroCreateKernel = "levelZeroCreateKernel";
static constexpr const char *kLevelZeroSetKernelArg = "levelZeroSetKernelArg";
static constexpr const char *kLevelZeroAddKernelDep = "levelZeroAddKernelDep";
static constexpr const char *kLevelZeroScheduleFunc = "levelZeroScheduleFunc";
static constexpr const char *kLevelZeroBarrier = "levelZeroBarrier";
static constexpr const char *kLevelZeroSubmit = "levelZeroSubmit";
static constexpr const char *kLevelZeroWait = "levelZeroWait";

namespace {

class ConvertCompToLevelZero
    : public ConvertCompToLevelZeroBase<ConvertCompToLevelZero> {
public:
  void runOnOperation();
  // TODO comp shall provide comp.submit for lower
};

void ConvertCompToLevelZero::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::OpBuilder builder(module.getOperation());
  mlir::Value execEnv;
  module.walk([&](mlir::Operation *op) {
    if (auto execEnvOp = mlir::dyn_cast_or_null<comp::CreateExecEnv>(op)) {
      execEnv = execEnvOp.getResult();
    } else if (auto waitOp = mlir::dyn_cast_or_null<comp::Wait>(op)) {
      builder.setInsertionPoint(waitOp.getOperation());
      builder.create<comp::Submit>(waitOp.getLoc(), execEnv);
    }
  });
  // Serialize SPIRV kernels.
  BinaryModulesMap modulesMap;
  if (mlir::failed(serializeSpirvKernels(module, modulesMap)))
    return signalPassFailure();
  // Populate conversion patterns.
  mlir::MLIRContext *context = &getContext();
  mlir::TypeConverter typeConverter, signatureConverter;
  mlir::OwningRewritePatternList patterns;
  populateCommonPatterns(context, typeConverter, signatureConverter, patterns);
  populateCompToLevelZeroPatterns(context, modulesMap, typeConverter, patterns);
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
  addLevelZeroFunctionDeclarations(module);
}

template <class Op>
struct ConvertCompToLevelZeroBasePattern : ConvertCompOpBasePattern<Op> {
  ConvertCompToLevelZeroBasePattern(mlir::TypeConverter &typeConverter,
                                    mlir::MLIRContext *context)
      : ConvertCompOpBasePattern<Op>(comp::ExecEnvRuntime::OpenCL,
                                     typeConverter, context) {}
};

/// Pattern for converting operation to llvm function call,
/// performing type conversions for results.
/// It can also handle variadic arguments when configured with
/// `varArg` and `nonVarArgs` constructor parameters.
template <class Op>
struct ConvertToFuncCallPattern : ConvertCompToLevelZeroBasePattern<Op> {
  ConvertToFuncCallPattern(mlir::StringRef funcName,
                           mlir::TypeConverter &typeConverter,
                           mlir::MLIRContext *context, bool varArg = false,
                           unsigned nonVarArgs = 0)
      : ConvertCompToLevelZeroBasePattern<Op>(typeConverter, context),
        funcName(funcName), varArg(varArg), nonVarArgs(nonVarArgs) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override;

  mlir::StringRef funcName;
  bool varArg;
  unsigned nonVarArgs;
};

using ConvertCreateExecEnv = ConvertToFuncCallPattern<comp::CreateExecEnv>;
using ConvertDestroyExecEnv = ConvertToFuncCallPattern<comp::DestroyExecEnv>;
using ConvertDealloc = ConvertToFuncCallPattern<comp::Dealloc>;
using ConvertScheduleBarrier = ConvertToFuncCallPattern<comp::ScheduleBarrier>;
using ConvertSubmit = ConvertToFuncCallPattern<comp::Submit>;
using ConvertWait = ConvertToFuncCallPattern<comp::Wait>;

/// Template pattern common for both comp::ScheduleRead and
/// comp::ScheduleWrite.
template <class Op>
struct ConvertScheduleReadWrite : ConvertCompToLevelZeroBasePattern<Op> {
  ConvertScheduleReadWrite(mlir::StringRef funcName,
                           mlir::TypeConverter &typeConverter,
                           mlir::MLIRContext *context)
      : ConvertCompToLevelZeroBasePattern<Op>(typeConverter, context),
        funcName(funcName) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override;

  mlir::StringRef funcName;
};

using ConvertScheduleRead = ConvertScheduleReadWrite<comp::ScheduleRead>;
using ConvertScheduleWrite = ConvertScheduleReadWrite<comp::ScheduleWrite>;

struct ConvertAlloc : ConvertCompToLevelZeroBasePattern<comp::Alloc> {
  using ConvertCompToLevelZeroBasePattern<
      comp::Alloc>::ConvertCompToLevelZeroBasePattern;

  mlir::LogicalResult
  matchAndRewrite(comp::Alloc op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

struct ConvertScheduleFunc
    : ConvertCompToLevelZeroBasePattern<comp::ScheduleFunc> {
  ConvertScheduleFunc(const BinaryModulesMap &modulesMap,
                      mlir::TypeConverter &typeConverter,
                      mlir::MLIRContext *context)
      : ConvertCompToLevelZeroBasePattern<comp::ScheduleFunc>(typeConverter,
                                                              context),
        modulesMap(modulesMap) {}

  mlir::LogicalResult
  matchAndRewrite(comp::ScheduleFunc op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override;

  const BinaryModulesMap &modulesMap;
};

} // namespace

void populateCompToLevelZeroPatterns(mlir::MLIRContext *context,
                                     const BinaryModulesMap &modulesMap,
                                     mlir::TypeConverter &typeConverter,
                                     mlir::OwningRewritePatternList &patterns) {
  // Populate type conversion patterns.
  LLVM::LLVMType llvmInt8Ptr = LLVM::LLVMType::getInt8PtrTy(context);
  typeConverter.addConversion(
      [=](comp::ExecEnvType execEnvType) -> mlir::Optional<mlir::Type> {
        if (execEnvType.getRuntime() != comp::ExecEnvRuntime::OpenCL)
          return llvm::None;
        return llvmInt8Ptr;
      });
  typeConverter.addConversion(
      [=](comp::EventType eventType) -> mlir::Optional<mlir::Type> {
        if (eventType.getRuntime() != comp::ExecEnvRuntime::OpenCL)
          return llvm::None;
        return llvmInt8Ptr;
      });
  // Populate operation conversion patterns.
  patterns.insert<ConvertCreateExecEnv>(kLevelZeroCreate, typeConverter,
                                        context);
  patterns.insert<ConvertDestroyExecEnv>(kLevelZeroDestroy, typeConverter,
                                         context);
  patterns.insert<ConvertDealloc>(kLevelZeroDealloc, typeConverter, context);
  patterns.insert<ConvertScheduleBarrier>(kLevelZeroBarrier, typeConverter,
                                          context,
                                          /*varArg=*/true, /*nonVarArgs=*/1);
  patterns.insert<ConvertSubmit>(kLevelZeroSubmit, typeConverter, context);
  patterns.insert<ConvertWait>(kLevelZeroWait, typeConverter, context,
                               /*varArg=*/true, /*nonVarArgs=*/0);

  patterns.insert<ConvertScheduleRead>(kLevelZeroRead, typeConverter, context);
  patterns.insert<ConvertScheduleWrite>(kLevelZeroWrite, typeConverter,
                                        context);

  patterns.insert<ConvertAlloc>(typeConverter, context);
  patterns.insert<ConvertScheduleFunc>(modulesMap, typeConverter, context);
}

void addLevelZeroFunctionDeclarations(mlir::ModuleOp &module) {
  mlir::Location loc = module.getLoc();
  mlir::OpBuilder builder(module.getBody()->getTerminator());
  mlir::MLIRContext *context = builder.getContext();
  LLVM::LLVMType llvmInt8Ptr = LLVM::LLVMType::getInt8PtrTy(context);
  LLVM::LLVMType llvmVoid = LLVM::LLVMType::getVoidTy(context);
  LLVM::LLVMType llvmInt32 = LLVM::LLVMType::getInt32Ty(context);

  if (!module.lookupSymbol(kLevelZeroCreate)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kLevelZeroCreate,
        LLVM::LLVMType::getFunctionTy(llvmInt8Ptr, {llvmInt8Ptr},
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kLevelZeroDestroy)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kLevelZeroDestroy,
        LLVM::LLVMType::getFunctionTy(llvmVoid, {llvmInt8Ptr},
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kLevelZeroAlloc)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kLevelZeroAlloc,
        LLVM::LLVMType::getFunctionTy(llvmInt8Ptr, {llvmInt8Ptr, llvmInt32},
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kLevelZeroDealloc)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kLevelZeroDealloc,
        LLVM::LLVMType::getFunctionTy(llvmVoid, {llvmInt8Ptr, llvmInt8Ptr},
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kLevelZeroRead)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kLevelZeroRead,
        LLVM::LLVMType::getFunctionTy(
            llvmInt8Ptr, {llvmInt8Ptr, llvmInt8Ptr, llvmInt8Ptr, llvmInt32},
            /*isVarArg=*/true));
  }
  if (!module.lookupSymbol(kLevelZeroWrite)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kLevelZeroWrite,
        LLVM::LLVMType::getFunctionTy(
            llvmInt8Ptr, {llvmInt8Ptr, llvmInt8Ptr, llvmInt8Ptr, llvmInt32},
            /*isVarArg=*/true));
  }
  if (!module.lookupSymbol(kLevelZeroCreateKernel)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kLevelZeroCreateKernel,
        LLVM::LLVMType::getFunctionTy(
            llvmInt8Ptr, {llvmInt8Ptr, llvmInt8Ptr, llvmInt32, llvmInt8Ptr},
            /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kLevelZeroSetKernelArg)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kLevelZeroSetKernelArg,
        LLVM::LLVMType::getFunctionTy(llvmVoid,
                                      {llvmInt8Ptr, llvmInt32, llvmInt8Ptr},
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kLevelZeroAddKernelDep)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kLevelZeroAddKernelDep,
        LLVM::LLVMType::getFunctionTy(llvmVoid, {llvmInt8Ptr, llvmInt8Ptr},
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kLevelZeroScheduleFunc)) {
    mlir::Type indexType = builder.getIndexType();
    builder.create<mlir::FuncOp>(
        loc, kLevelZeroScheduleFunc,
        mlir::FunctionType::get({llvmInt8Ptr, llvmInt8Ptr, indexType, indexType,
                                 indexType, indexType, indexType, indexType},
                                {llvmInt8Ptr}, context));
  }
  if (!module.lookupSymbol(kLevelZeroBarrier)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kLevelZeroBarrier,
        LLVM::LLVMType::getFunctionTy(llvmInt8Ptr, {llvmInt8Ptr, llvmInt32},
                                      /*isVarArg=*/true));
  }
  if (!module.lookupSymbol(kLevelZeroSubmit)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kLevelZeroSubmit,
        LLVM::LLVMType::getFunctionTy(llvmVoid, {llvmInt8Ptr},
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kLevelZeroWait)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kLevelZeroWait,
        LLVM::LLVMType::getFunctionTy(llvmVoid, {llvmInt32},
                                      /*isVarArg=*/true));
  }
}

template <class Op>
mlir::LogicalResult ConvertToFuncCallPattern<Op>::matchAndRewrite(
    Op op, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (!this->isMatchingRuntime(op))
    return mlir::failure();

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

template <class Op>
mlir::LogicalResult ConvertScheduleReadWrite<Op>::matchAndRewrite(
    Op op, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (!this->isMatchingRuntime(op))
    return mlir::failure();

  constexpr unsigned nonVarArgs = 3;
  mlir::SmallVector<mlir::Value, nonVarArgs + 2> castOperands(
      operands.begin(), operands.begin() + nonVarArgs);

  // Convert host memref to pointer.
  mlir::Value hostPtr =
      this->materializeConversion(rewriter, op.getLoc(), operands[0]);
  if (!hostPtr)
    return mlir::failure();
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

  mlir::Type llvmResultType = convertType(op.getType());
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(
      op.getOperation(), mlir::ArrayRef<mlir::Type>{llvmResultType},
      rewriter.getSymbolRefAttr(kLevelZeroAlloc), castOperands);
  return mlir::success();
}

mlir::LogicalResult ConvertScheduleFunc::matchAndRewrite(
    comp::ScheduleFunc op, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (!isMatchingRuntime(op))
    return mlir::failure();

  mlir::Location loc = op.getLoc();
  auto launchOp = mlir::cast<gpu::LaunchFuncOp>(op.body().front().front());
  std::string binaryName = launchOp.getKernelModuleName().str();
  std::string kernelName = launchOp.getKernelName().str();
  mlir::Type llvmEventType = convertType(op.getType());
  LLVM::LLVMType llvmKernelType =
      LLVM::LLVMType::getInt8PtrTy(rewriter.getContext());

  // Create kernel from serialized binary.
  if (modulesMap.count(binaryName) == 0)
    return mlir::failure();
  if (modulesMap.at(binaryName).kernelsNameMap.count(kernelName) == 0)
    return mlir::failure();

  mlir::Value binaryPtr, binaryBytes;
  getPtrToBinaryModule(rewriter, loc, modulesMap.at(binaryName), binaryPtr,
                       binaryBytes);
  mlir::Value namePtr = getPtrToGlobalString(
      rewriter, loc, modulesMap.at(binaryName).kernelsNameMap.at(kernelName));

  auto createCall = rewriter.create<LLVM::CallOp>(
      loc, mlir::ArrayRef<mlir::Type>(llvmKernelType),
      rewriter.getSymbolRefAttr(kLevelZeroCreateKernel),
      mlir::ArrayRef<mlir::Value>{operands[0], binaryPtr, binaryBytes,
                                  namePtr});
  mlir::Value kernel = createCall.getResult(0);

  // Set kernel arguments.
  for (unsigned argI = 0; argI < launchOp.getNumKernelOperands(); ++argI) {
    mlir::Type llvmInt32Type =
        LLVM::LLVMType::getInt32Ty(rewriter.getContext());
    mlir::Value argIndex = rewriter.create<LLVM::ConstantOp>(
        loc, llvmInt32Type, rewriter.getI32IntegerAttr(argI));
    mlir::Value remappedArg =
        rewriter.getRemappedValue(launchOp.getKernelOperand(argI));

    rewriter.create<LLVM::CallOp>(
        loc, mlir::ArrayRef<mlir::Type>{},
        rewriter.getSymbolRefAttr(kLevelZeroSetKernelArg),
        mlir::ArrayRef<mlir::Value>{kernel, argIndex, remappedArg});
  }
  // Set event dependencies. This is done with separate functions
  // on kernel as opposed to variadic argument in final function,
  // because dispatch sizes are index types prohibiting use of
  // llvm function and variadic arguments.
  for (mlir::Value event : operands.slice(1)) {
    rewriter.create<LLVM::CallOp>(
        loc, mlir::ArrayRef<mlir::Type>{},
        rewriter.getSymbolRefAttr(kLevelZeroAddKernelDep),
        mlir::ArrayRef<mlir::Value>{kernel, event});
  }

  auto gridSize = launchOp.getGridSizeOperandValues();
  auto blockSize = launchOp.getBlockSizeOperandValues();
  // OpenCL takes as global work size number of blocks times block size,
  // so multiplications are needed.
  auto globalX = rewriter.create<mlir::MulIOp>(loc, gridSize.x, blockSize.x);
  auto globalY = rewriter.create<mlir::MulIOp>(loc, gridSize.y, blockSize.y);
  auto globalZ = rewriter.create<mlir::MulIOp>(loc, gridSize.z, blockSize.z);

  mlir::SmallVector<mlir::Value, 8> scheduleArgs{
      operands[0], kernel,      globalX,     globalY,
      globalZ,     blockSize.x, blockSize.y, blockSize.z};
  rewriter.replaceOpWithNewOp<mlir::CallOp>(
      op.getOperation(), mlir::ArrayRef<mlir::Type>{llvmEventType},
      rewriter.getSymbolRefAttr(kLevelZeroScheduleFunc), scheduleArgs);

  return mlir::success();
}

std::unique_ptr<mlir::Pass> createConvertCompToLevelZeroPass() {
  return std::make_unique<ConvertCompToLevelZero>();
}

} // namespace pmlc::conversion::comp_to_llvm
