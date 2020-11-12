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

static constexpr const char *kOclCreate = "oclCreate";
static constexpr const char *kOclDestroy = "oclDestroy";
static constexpr const char *kOclAlloc = "oclAlloc";
static constexpr const char *kOclDealloc = "oclDealloc";
static constexpr const char *kOclRead = "oclRead";
static constexpr const char *kOclWrite = "oclWrite";
static constexpr const char *kOclCreateKernel = "oclCreateKernel";
static constexpr const char *kOclDestroyKernel = "oclDestroyKernel";
static constexpr const char *kOclSetKernelArg = "oclSetKernelArg";
static constexpr const char *kOclScheduleFunc = "oclScheduleFunc";
static constexpr const char *kOclBarrier = "oclBarrier";
static constexpr const char *kOclSubmit = "oclSubmit";
static constexpr const char *kOclWait = "oclWait";

namespace {

class ConvertCompToOcl : public ConvertCompToOclBase<ConvertCompToOcl> {
public:
  void runOnOperation();
};

void ConvertCompToOcl::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  // Serialize SPIRV kernels.
  BinaryModulesMap modulesMap;
  if (mlir::failed(serializeSpirvKernels(module, modulesMap)))
    return signalPassFailure();
  // Populate conversion patterns.
  mlir::MLIRContext *context = &getContext();
  mlir::LLVMTypeConverter typeConverter{context};
  mlir::OwningRewritePatternList patterns;
  populateCommonPatterns(context, typeConverter, patterns);
  populateCompToOclPatterns(context, modulesMap, typeConverter, patterns);
  // Set conversion target.
  mlir::ConversionTarget target(*context);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalDialect<mlir::StandardOpsDialect>();
  target.addIllegalDialect<comp::COMPDialect>();
  target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) -> bool {
    return typeConverter.isSignatureLegal(op.getType());
  });
  if (mlir::failed(mlir::applyPartialConversion(module, target, patterns)))
    signalPassFailure();
  // Insert runtime function declarations.
  addCommonFunctionDeclarations(module);
  addOclFunctionDeclarations(module, typeConverter);
}

template <class Op>
struct ConvertCompToOclBasePattern : ConvertCompOpBasePattern<Op> {
  ConvertCompToOclBasePattern(mlir::TypeConverter &typeConverter,
                              mlir::MLIRContext *context)
      : ConvertCompOpBasePattern<Op>(comp::ExecEnvRuntime::OpenCL,
                                     typeConverter, context) {}
};

/// Pattern for converting operation to llvm function call,
/// performing type conversions for results.
/// It can also handle variadic arguments when configured with
/// `varArg` and `nonVarArgs` constructor parameters.
template <class Op>
struct ConvertToFuncCallPattern : ConvertCompToOclBasePattern<Op> {
  ConvertToFuncCallPattern(mlir::StringRef funcName,
                           mlir::TypeConverter &typeConverter,
                           mlir::MLIRContext *context, bool varArg = false,
                           unsigned nonVarArgs = 0)
      : ConvertCompToOclBasePattern<Op>(typeConverter, context),
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
using ConvertScheduleBarrier = ConvertToFuncCallPattern<comp::ScheduleBarrier>;
using ConvertSubmit = ConvertToFuncCallPattern<comp::Submit>;
using ConvertWait = ConvertToFuncCallPattern<comp::Wait>;

/// Template pattern common for both comp::ScheduleRead and
/// comp::ScheduleWrite.
template <class Op>
struct ConvertScheduleReadWrite final
    : public mlir::ConvertOpToLLVMPattern<Op> {
  using mlir::ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final;

private:
  static mlir::StringRef funcName();
};

template <>
mlir::StringRef ConvertScheduleReadWrite<comp::ScheduleRead>::funcName() {
  return kOclRead;
}

template <>
mlir::StringRef ConvertScheduleReadWrite<comp::ScheduleWrite>::funcName() {
  return kOclWrite;
}

using ConvertScheduleRead = ConvertScheduleReadWrite<comp::ScheduleRead>;
using ConvertScheduleWrite = ConvertScheduleReadWrite<comp::ScheduleWrite>;

struct ConvertAlloc final : mlir::ConvertOpToLLVMPattern<comp::Alloc> {
  using ConvertOpToLLVMPattern<comp::Alloc>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final;
};

struct ConvertDealloc final : mlir::ConvertOpToLLVMPattern<comp::Dealloc> {
  using ConvertOpToLLVMPattern<comp::Dealloc>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final;
};

struct ConvertCreateKernel final
    : mlir::ConvertOpToLLVMPattern<comp::CreateKernel> {
  ConvertCreateKernel(const BinaryModulesMap &modulesMap,
                      mlir::LLVMTypeConverter &typeConverter)
      : mlir::ConvertOpToLLVMPattern<comp::CreateKernel>(typeConverter),
        modulesMap(modulesMap) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final;

  const BinaryModulesMap &modulesMap;
};

struct ConvertDestroyKernel final
    : mlir::ConvertOpToLLVMPattern<comp::DestroyKernel> {
  using mlir::ConvertOpToLLVMPattern<
      comp::DestroyKernel>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final;
};

struct ConvertScheduleCompute final
    : mlir::ConvertOpToLLVMPattern<comp::ScheduleCompute> {
  using mlir::ConvertOpToLLVMPattern<
      comp::ScheduleCompute>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final;
};

} // namespace

void populateCompToOclPatterns(mlir::MLIRContext *context,
                               const BinaryModulesMap &modulesMap,
                               mlir::LLVMTypeConverter &typeConverter,
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
  typeConverter.addConversion(
      [=](comp::KernelType kernelType) -> mlir::Optional<mlir::Type> {
        return llvmInt8Ptr;
      });

  // TODO: Converting index->LLVM integer seems like something that the
  //       LLVMTypeConverter should be handling automatically.  Currently,
  //       though, it's materializing indices via a DialectCastOp, which
  //       explicitly doesn't work on indices.  So we add an explicit
  //       materialization, and hope this gets fixed in the standard->llvm
  //       conversion logic (which does have a TODO for it).
  typeConverter.addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> mlir::Optional<mlir::Value> {
        if (inputs.size() != 1) {
          return llvm::None;
        }
        mlir::Value value = inputs[0];
        if (!value.getType().isIndex()) {
          return llvm::None;
        }
        auto asInt = builder.create<mlir::IndexCastOp>(
            loc, builder.getIntegerType(typeConverter.getIndexTypeBitwidth()),
            value);
        return builder.create<LLVM::DialectCastOp>(loc, resultType, asInt)
            .getResult();
      });

  // Populate operation conversion patterns.
  patterns.insert<ConvertCreateExecEnv>(kOclCreate, typeConverter, context);
  patterns.insert<ConvertDestroyExecEnv>(kOclDestroy, typeConverter, context);
  patterns.insert<ConvertDealloc>(typeConverter);
  patterns.insert<ConvertScheduleBarrier>(kOclBarrier, typeConverter, context,
                                          /*varArg=*/true, /*nonVarArgs=*/1);
  patterns.insert<ConvertSubmit>(kOclSubmit, typeConverter, context);
  patterns.insert<ConvertWait>(kOclWait, typeConverter, context,
                               /*varArg=*/true, /*nonVarArgs=*/0);

  patterns.insert<ConvertScheduleRead>(typeConverter);
  patterns.insert<ConvertScheduleWrite>(typeConverter);

  patterns.insert<ConvertAlloc>(typeConverter);

  patterns.insert<ConvertCreateKernel>(modulesMap, typeConverter);
  patterns.insert<ConvertDestroyKernel>(typeConverter);

  patterns.insert<ConvertScheduleCompute>(typeConverter);
}

void addOclFunctionDeclarations(mlir::ModuleOp &module,
                                mlir::LLVMTypeConverter &typeConverter) {
  mlir::Location loc = module.getLoc();
  mlir::OpBuilder builder(module.getBody()->getTerminator());
  mlir::MLIRContext *context = builder.getContext();
  LLVM::LLVMType llvmInt8Ptr = LLVM::LLVMType::getInt8PtrTy(context);
  LLVM::LLVMType llvmVoid = LLVM::LLVMType::getVoidTy(context);
  LLVM::LLVMType llvmInt32 = LLVM::LLVMType::getInt32Ty(context);
  LLVM::LLVMType llvmIndex = typeConverter.getIndexType();

  if (!module.lookupSymbol(kOclCreate)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclCreate,
        LLVM::LLVMType::getFunctionTy(llvmInt8Ptr, {llvmInt8Ptr},
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kOclDestroy)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclDestroy,
        LLVM::LLVMType::getFunctionTy(llvmVoid, {llvmInt8Ptr},
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kOclAlloc)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclAlloc,
        LLVM::LLVMType::getFunctionTy(llvmInt8Ptr, {llvmInt8Ptr, llvmIndex},
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kOclDealloc)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclDealloc,
        LLVM::LLVMType::getFunctionTy(llvmVoid, {llvmInt8Ptr, llvmInt8Ptr},
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kOclRead)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclRead,
        LLVM::LLVMType::getFunctionTy(
            llvmInt8Ptr, {llvmInt8Ptr, llvmInt8Ptr, llvmInt8Ptr, llvmIndex},
            /*isVarArg=*/true));
  }
  if (!module.lookupSymbol(kOclWrite)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclWrite,
        LLVM::LLVMType::getFunctionTy(
            llvmInt8Ptr, {llvmInt8Ptr, llvmInt8Ptr, llvmInt8Ptr, llvmIndex},
            /*isVarArg=*/true));
  }
  if (!module.lookupSymbol(kOclCreateKernel)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclCreateKernel,
        LLVM::LLVMType::getFunctionTy(
            llvmInt8Ptr, {llvmInt8Ptr, llvmInt8Ptr, llvmInt32, llvmInt8Ptr},
            /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kOclDestroyKernel)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclDestroyKernel,
        LLVM::LLVMType::getFunctionTy(llvmVoid, {llvmInt8Ptr, llvmInt8Ptr},
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kOclSetKernelArg)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclSetKernelArg,
        LLVM::LLVMType::getFunctionTy(llvmVoid,
                                      {llvmInt8Ptr, llvmIndex, llvmInt8Ptr},
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kOclScheduleFunc)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclScheduleFunc,
        LLVM::LLVMType::getFunctionTy(llvmInt8Ptr,
                                      {llvmInt8Ptr, llvmInt8Ptr, llvmIndex,
                                       llvmIndex, llvmIndex, llvmIndex,
                                       llvmIndex, llvmIndex, llvmIndex},
                                      /*isVarArg=*/true));
  }
  if (!module.lookupSymbol(kOclBarrier)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclBarrier,
        LLVM::LLVMType::getFunctionTy(llvmInt8Ptr, {llvmInt8Ptr, llvmInt32},
                                      /*isVarArg=*/true));
  }
  if (!module.lookupSymbol(kOclSubmit)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclSubmit,
        LLVM::LLVMType::getFunctionTy(llvmVoid, {llvmInt8Ptr},
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kOclWait)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclWait,
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
    mlir::Operation *opPtr, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto op = mlir::cast<Op>(opPtr);
  auto loc = op.getLoc();

  constexpr unsigned nonVarArgs = 3;
  mlir::SmallVector<mlir::Value, nonVarArgs + 2> callOperands(
      operands.begin(), operands.begin() + nonVarArgs);

  // Extract the host memref memory pointer.
  callOperands[0] = hostMemrefToMem(rewriter, loc, operands[0]);

  // Extract the device memref memory pointer.
  callOperands[1] = deviceMemrefToMem(rewriter, loc, operands[1]);

  // Add event dependencies as variadic operands.
  mlir::Value eventsCnt = this->createIndexConstant(
      rewriter, op.getLoc(), operands.size() - nonVarArgs);
  callOperands.push_back(eventsCnt);
  callOperands.insert(callOperands.end(), operands.begin() + nonVarArgs,
                      operands.end());

  mlir::Type llvmEventType = this->typeConverter.convertType(op.getType());
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(
      op.getOperation(), mlir::ArrayRef<mlir::Type>{llvmEventType},
      rewriter.getSymbolRefAttr(funcName()), callOperands);
  return mlir::success();
}

mlir::LogicalResult
ConvertAlloc::matchAndRewrite(mlir::Operation *opPtr,
                              mlir::ArrayRef<mlir::Value> operands,
                              mlir::ConversionPatternRewriter &rewriter) const {
  auto op = mlir::cast<comp::Alloc>(opPtr);

  mlir::Location loc = op.getLoc();
  mlir::MemRefType resultType = op.getType().cast<mlir::MemRefType>();

  // Figure out the amount of memory we need to allocate.
  mlir::SmallVector<mlir::Value, 4> sizes;
  getMemRefDescriptorSizes(loc, resultType, {}, rewriter, sizes);
  auto sizeToAlloc = getCumulativeSizeInBytes(loc, resultType.getElementType(),
                                              sizes, rewriter);

  // Build the call to allocate memory on the device.
  auto alloc = rewriter.create<LLVM::CallOp>(
      loc, getVoidPtrType(), rewriter.getSymbolRefAttr(kOclAlloc),
      mlir::ValueRange{operands[0], sizeToAlloc});
  mlir::Value memRaw = alloc.getResult(0);
  auto targetType = typeConverter.convertType(resultType.getElementType())
                        .dyn_cast_or_null<LLVM::LLVMType>();
  if (!targetType) {
    return mlir::failure();
  }
  mlir::Value memTyped =
      rewriter.create<LLVM::BitcastOp>(loc, targetType.getPointerTo(), memRaw);
  mlir::Value memOnDev = rewriter.create<LLVM::AddrSpaceCastOp>(
      loc, LLVM::LLVMPointerType::get(targetType, resultType.getMemorySpace()),
      memTyped);

  // Build a memref descriptor for the result.
  auto memref = mlir::MemRefDescriptor::fromStaticShape(
      rewriter, loc, typeConverter, resultType, memOnDev);

  rewriter.replaceOp(op, {memref});
  return mlir::success();
}

mlir::LogicalResult ConvertDealloc::matchAndRewrite(
    mlir::Operation *opPtr, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto op = mlir::cast<comp::Dealloc>(opPtr);

  // Build the dealloc call.
  rewriter.create<LLVM::CallOp>(
      op.getLoc(), mlir::TypeRange{}, rewriter.getSymbolRefAttr(kOclDealloc),
      mlir::ValueRange{operands[0],
                       deviceMemrefToMem(rewriter, op.getLoc(), operands[1])});

  rewriter.eraseOp(op);
  return mlir::success();
}

mlir::LogicalResult ConvertCreateKernel::matchAndRewrite(
    mlir::Operation *opPtr, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto op = mlir::cast<comp::CreateKernel>(opPtr);
  auto loc = op.getLoc();

  auto binaryName = op.kernelFunc().getRootReference().str();
  auto kernelName = op.kernelFunc().getLeafReference().str();

  if (modulesMap.count(binaryName) == 0) {
    return mlir::failure();
  }
  if (modulesMap.at(binaryName).kernelsNameMap.count(kernelName) == 0) {
    return mlir::failure();
  }

  mlir::Value binaryPtr, binaryBytes;
  getPtrToBinaryModule(rewriter, loc, modulesMap.at(binaryName), binaryPtr,
                       binaryBytes);
  mlir::Value namePtr = getPtrToGlobalString(
      rewriter, loc, modulesMap.at(binaryName).kernelsNameMap.at(kernelName));

  rewriter.replaceOpWithNewOp<LLVM::CallOp>(
      op, getVoidPtrType(), rewriter.getSymbolRefAttr(kOclCreateKernel),
      mlir::ArrayRef<mlir::Value>{operands[0], binaryPtr, binaryBytes,
                                  namePtr});

  return mlir::success();
}

mlir::LogicalResult ConvertDestroyKernel::matchAndRewrite(
    mlir::Operation *opPtr, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(
      opPtr, mlir::TypeRange{}, rewriter.getSymbolRefAttr(kOclDestroyKernel),
      operands);
  return mlir::success();
}

mlir::LogicalResult ConvertScheduleCompute::matchAndRewrite(
    mlir::Operation *opPtr, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto op = mlir::cast<comp::ScheduleCompute>(opPtr);
  auto loc = op.getLoc();

  // Set kernel arguments.
  auto kernel = rewriter.getRemappedValue(op.kernel());
  auto setKernelArg = rewriter.getSymbolRefAttr(kOclSetKernelArg);
  for (auto bufferIdx : llvm::enumerate(op.buffers())) {
    rewriter.create<LLVM::CallOp>(
        loc, mlir::TypeRange{}, setKernelArg,
        mlir::ValueRange{
            kernel, createIndexConstant(rewriter, loc, bufferIdx.index()),
            deviceMemrefToMem(rewriter, loc,
                              rewriter.getRemappedValue(bufferIdx.value()))});
  }

  // OpenCL takes as global work size number of blocks times block size,
  // so multiplications are needed.
  auto globalX =
      rewriter.create<mlir::MulIOp>(loc, op.gridSizeX(), op.blockSizeX());
  auto globalY =
      rewriter.create<mlir::MulIOp>(loc, op.gridSizeY(), op.blockSizeY());
  auto globalZ =
      rewriter.create<mlir::MulIOp>(loc, op.gridSizeZ(), op.blockSizeZ());

  mlir::SmallVector<mlir::Value, 16> callArgs{
      rewriter.getRemappedValue(op.execEnv()),
      kernel,
      indexToInt(rewriter, loc, typeConverter, globalX),
      indexToInt(rewriter, loc, typeConverter, globalY),
      indexToInt(rewriter, loc, typeConverter, globalZ),
      indexToInt(rewriter, loc, typeConverter, op.blockSizeX()),
      indexToInt(rewriter, loc, typeConverter, op.blockSizeY()),
      indexToInt(rewriter, loc, typeConverter, op.blockSizeZ()),
      createIndexConstant(rewriter, loc, op.depEvents().size())};
  for (auto dep : op.depEvents()) {
    callArgs.emplace_back(rewriter.getRemappedValue(dep));
  }

  rewriter.replaceOpWithNewOp<LLVM::CallOp>(
      op.getOperation(), getVoidPtrType(),
      rewriter.getSymbolRefAttr(kOclScheduleFunc), callArgs);

  return mlir::success();
}

std::unique_ptr<mlir::Pass> createConvertCompToOclPass() {
  return std::make_unique<ConvertCompToOcl>();
}

} // namespace pmlc::conversion::comp_to_llvm
