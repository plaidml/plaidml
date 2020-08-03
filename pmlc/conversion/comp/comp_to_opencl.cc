// Copyright 2020, Intel Corporation
#include <iostream>
#include <map>
#include <string>

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/comp/pass_detail.h"
#include "pmlc/dialect/comp/ir/dialect.h"

namespace pmlc::conversion::comp {

namespace comp = pmlc::dialect::comp;
namespace spirv = mlir::spirv;
namespace LLVM = mlir::LLVM;
namespace gpu = mlir::gpu;

static constexpr const char *kSpirvBinPrefix = "_ocl_spirv_bin_";

static constexpr const char *kOclInit = "oclInit";
static constexpr const char *kOclDeinit = "oclDeinit";
static constexpr const char *kOclCreateKernel = "oclCreateKernel";
static constexpr const char *kOclEnqueueKernel = "oclEnqueueKernel";
static constexpr const char *kOclSetKernelArg = "oclSetKernelArg";
static constexpr const char *kOclAllocBuffer = "oclAllocBuffer";
static constexpr const char *kOclDeallocBuffer = "oclDeallocBuffer";
static constexpr const char *kOclEnqueueRead = "oclEnqueueRead";
static constexpr const char *kOclEnqueueWrite = "oclEnqueueWrite";
static constexpr const char *kOclGroupEvents = "oclGroupEvents";
static constexpr const char *kOclWait = "oclWait";

struct HelperCache {
  void init(mlir::MLIRContext &context) {
    llvmDialect = context.getRegisteredDialect<LLVM::LLVMDialect>();
    llvmVoidType = LLVM::LLVMType::getVoidTy(llvmDialect);
    llvmPointerType = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    llvmInt8Type = LLVM::LLVMType::getInt8Ty(llvmDialect);
    llvmInt32Type = LLVM::LLVMType::getInt32Ty(llvmDialect);
    llvmInt64Type = LLVM::LLVMType::getInt64Ty(llvmDialect);
    mlirIndexType = mlir::IndexType::get(&context);
  }

  LLVM::LLVMDialect *llvmDialect;
  LLVM::LLVMType llvmVoidType;
  LLVM::LLVMType llvmPointerType;
  LLVM::LLVMType llvmInt8Type;
  LLVM::LLVMType llvmInt32Type;
  LLVM::LLVMType llvmInt64Type;
  mlir::Type mlirIndexType;
};

class ConvertCompToOpenCl
    : public ConvertCompToOpenClBase<ConvertCompToOpenCl> {
public:
  void runOnOperation() override;

  mlir::LogicalResult serializeSpirvKernel(spirv::ModuleOp moduleOp);
  mlir::LogicalResult convertCompOps();
  void declareOpenClFunctions();

  HelperCache helperCache;

  LLVM::LLVMDialect *getLLVMDialect() { return helperCache.llvmDialect; }
  LLVM::LLVMType getLLVMVoidType() { return helperCache.llvmVoidType; }
  LLVM::LLVMType getLLVMPointerType() { return helperCache.llvmPointerType; }
  LLVM::LLVMType getLLVMInt8Type() { return helperCache.llvmInt8Type; }
  LLVM::LLVMType getLLVMInt32Type() { return helperCache.llvmInt32Type; }
  mlir::Type getMLIRIndexType() { return helperCache.mlirIndexType; }

  struct BinaryKernelInfo {
    LLVM::GlobalOp symbol;
    size_t bytes;
  };

  using KernelsMap = std::map<std::string, BinaryKernelInfo>;
  KernelsMap kernelsMap;
};

void ConvertCompToOpenCl::runOnOperation() {
  helperCache.init(getContext());

  auto moduleOp = getOperation();
  auto serializationWalk =
      moduleOp.walk([&](spirv::ModuleOp op) -> mlir::WalkResult {
        return serializeSpirvKernel(op);
      });
  if (serializationWalk.wasInterrupted())
    signalPassFailure();
  if (mlir::failed(convertCompOps()))
    signalPassFailure();
  declareOpenClFunctions();
}

mlir::LogicalResult
ConvertCompToOpenCl::serializeSpirvKernel(spirv::ModuleOp moduleOp) {
  // Serialize kernel
  mlir::SmallVector<uint32_t, 0> moduleBinary;
  if (mlir::failed(spirv::serialize(moduleOp, moduleBinary)))
    return mlir::failure();

  // Add it to current top module
  auto gpuModule =
      mlir::dyn_cast<gpu::GPUModuleOp>(moduleOp.getOperation()->getNextNode());
  if (!gpuModule)
    return mlir::failure(); // This is a hack due to how kernel outlining works
  std::string gpuModuleName = static_cast<std::string>(gpuModule.getName());
  std::string binaryName = kSpirvBinPrefix + gpuModuleName;

  auto topModule = getOperation();
  auto llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
  mlir::OpBuilder builder(topModule.getBodyRegion());

  auto binaryType =
      LLVM::LLVMType::getArrayTy(getLLVMInt8Type(), moduleBinary.size() * 4);

  auto binaryOp = builder.create<LLVM::GlobalOp>(
      moduleOp.getLoc(), binaryType,
      /*isConstant=*/true, LLVM::Linkage::Internal, binaryName,
      builder.getStringAttr({reinterpret_cast<char *>(moduleBinary.data()),
                             moduleBinary.size() * 4}));

  // Update kernels map
  kernelsMap[gpuModuleName] = {binaryOp, moduleBinary.size() * 4};

  return mlir::success();
}

void ConvertCompToOpenCl::declareOpenClFunctions() {
  auto module = getOperation();
  auto loc = module.getLoc();
  mlir::OpBuilder builder(module.getBody()->getTerminator());

  if (!module.lookupSymbol(kOclInit)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclInit,
        LLVM::LLVMType::getFunctionTy(getLLVMPointerType(), {},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kOclDeinit)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclDeinit,
        LLVM::LLVMType::getFunctionTy(getLLVMVoidType(), {getLLVMPointerType()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kOclCreateKernel)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclCreateKernel,
        LLVM::LLVMType::getFunctionTy(
            getLLVMPointerType(),
            {getLLVMPointerType(), getLLVMPointerType(), getLLVMInt32Type()},
            /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kOclEnqueueKernel)) {
    auto &ctx = getContext();

    builder.create<mlir::FuncOp>(
        loc, kOclEnqueueKernel,
        mlir::FunctionType::get(
            mlir::ArrayRef<mlir::Type>{
                getLLVMPointerType(), getLLVMPointerType(), getMLIRIndexType(),
                getMLIRIndexType(), getMLIRIndexType(), getMLIRIndexType(),
                getMLIRIndexType(), getMLIRIndexType(), getLLVMPointerType()},
            {getLLVMPointerType()}, &ctx),
        mlir::ArrayRef<std::pair<mlir::Identifier, mlir::Attribute>>());
  }

  if (!module.lookupSymbol(kOclSetKernelArg)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclSetKernelArg,
        LLVM::LLVMType::getFunctionTy(getLLVMVoidType(),
                                      {getLLVMPointerType(),
                                       getLLVMPointerType(), getLLVMInt32Type(),
                                       getLLVMPointerType()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kOclAllocBuffer)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclAllocBuffer,
        LLVM::LLVMType::getFunctionTy(
            getLLVMPointerType(),
            {getLLVMPointerType(), getLLVMInt32Type(), getLLVMPointerType()},
            /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kOclDeallocBuffer)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclDeallocBuffer,
        LLVM::LLVMType::getFunctionTy(getLLVMVoidType(), {getLLVMPointerType()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kOclEnqueueRead)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclEnqueueRead,
        LLVM::LLVMType::getFunctionTy(
            getLLVMPointerType(),
            {getLLVMPointerType(), getLLVMPointerType(), getLLVMPointerType(),
             getLLVMPointerType()},
            /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kOclEnqueueWrite)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclEnqueueWrite,
        LLVM::LLVMType::getFunctionTy(
            getLLVMPointerType(),
            {getLLVMPointerType(), getLLVMPointerType(), getLLVMPointerType(),
             getLLVMPointerType()},
            /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kOclGroupEvents)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclGroupEvents,
        LLVM::LLVMType::getFunctionTy(getLLVMPointerType(),
                                      {getLLVMInt32Type()},
                                      /*isVarArg=*/true));
  }

  if (!module.lookupSymbol(kOclWait)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kOclWait,
        LLVM::LLVMType::getFunctionTy(getLLVMVoidType(), {getLLVMPointerType()},
                                      /*isVarArg=*/false));
  }
}

struct CompLLVMTypeConverter : public mlir::TypeConverter {
  explicit CompLLVMTypeConverter(HelperCache &helperCache)
      : helperCache(helperCache) {
    addConversion(
        [&](comp::ExecEnvType) { return helperCache.llvmPointerType; });
    addConversion([&](comp::EventType) { return helperCache.llvmPointerType; });
    addConversion([&](mlir::MemRefType mem) {
      return helperCache.llvmInt8Type.getPointerTo(mem.getMemorySpace());
    });

    // addMaterialization([&](mlir::PatternRewriter& rewriter,
    //                        mlir::MemRefType memType,
    //                        mlir::ValueRange memVals,
    //                        mlir::Location loc) {
    //   mlir::LLVMTypeConverter llvmConverter;
    //   auto structMemType = llvmConverter.convertType(memType);
    //   auto structMemRef = rewriter.create<LLVM::DialectCastOp>(loc,
    //   structMemType, memVals[0]); auto bufferDesc =
    //   mlir::MemRefDescriptor(structMemRef); auto flatMemAllocPtr =
    //   bufferDesc.allocatedPtr(rewriter, loc); memPtr =
    //   rewriter.create<LLVM::BitcastOp>(
    //     loc, convertType(memType), flatMemAllocPtr
    //   );
    //   return memPtr;
    // });
  }

  HelperCache &helperCache;
};

struct ConvertCreateExecEnv
    : public mlir::OpConversionPattern<comp::CreateExecEnv> {
  ConvertCreateExecEnv(mlir::MLIRContext *context, HelperCache &helperCache)
      : mlir::OpConversionPattern<comp::CreateExecEnv>(context),
        helperCache(helperCache) {}

  mlir::LogicalResult
  matchAndRewrite(comp::CreateExecEnv op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const;

  HelperCache &helperCache;
};

struct ConvertDestroyExecEnv
    : public mlir::OpConversionPattern<comp::DestroyExecEnv> {
  ConvertDestroyExecEnv(mlir::MLIRContext *context, HelperCache &helperCache)
      : mlir::OpConversionPattern<comp::DestroyExecEnv>(context),
        helperCache(helperCache) {}

  mlir::LogicalResult
  matchAndRewrite(comp::DestroyExecEnv op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const;

  HelperCache &helperCache;
};

struct ConvertAlloc : public mlir::OpConversionPattern<comp::Alloc> {
  ConvertAlloc(mlir::MLIRContext *context, CompLLVMTypeConverter &converter,
               HelperCache &helperCache)
      : mlir::OpConversionPattern<comp::Alloc>(converter, context),
        helperCache(helperCache) {}

  mlir::LogicalResult
  matchAndRewrite(comp::Alloc op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const;

  HelperCache &helperCache;
};

struct ConvertDealloc : public mlir::OpConversionPattern<comp::Dealloc> {
  ConvertDealloc(mlir::MLIRContext *context, HelperCache &helperCache)
      : mlir::OpConversionPattern<comp::Dealloc>(context),
        helperCache(helperCache) {}

  mlir::LogicalResult
  matchAndRewrite(comp::Dealloc op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const;

  HelperCache &helperCache;
};

struct ConvertScheduleRead
    : public mlir::OpConversionPattern<comp::ScheduleRead> {
  ConvertScheduleRead(mlir::MLIRContext *context,
                      CompLLVMTypeConverter &converter,
                      HelperCache &helperCache)
      : mlir::OpConversionPattern<comp::ScheduleRead>(converter, context),
        helperCache(helperCache) {}

  mlir::LogicalResult
  matchAndRewrite(comp::ScheduleRead op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const;

  HelperCache &helperCache;
};

struct ConvertWait : public mlir::OpConversionPattern<comp::Wait> {
  ConvertWait(mlir::MLIRContext *context, HelperCache &helperCache)
      : mlir::OpConversionPattern<comp::Wait>(context),
        helperCache(helperCache) {}

  mlir::LogicalResult
  matchAndRewrite(comp::Wait op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const;

  HelperCache &helperCache;
};

struct ConvertScheduleFunc
    : public mlir::OpConversionPattern<comp::ScheduleFunc> {
  ConvertScheduleFunc(mlir::MLIRContext *context, HelperCache &helperCache,
                      ConvertCompToOpenCl::KernelsMap &kernelsMap)
      : mlir::OpConversionPattern<comp::ScheduleFunc>(context),
        helperCache(helperCache), kernelsMap(kernelsMap) {}

  mlir::LogicalResult
  matchAndRewrite(comp::ScheduleFunc op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const;

  HelperCache &helperCache;
  ConvertCompToOpenCl::KernelsMap &kernelsMap;
};

struct ConvertGroupEvents
    : public mlir::OpConversionPattern<comp::GroupEvents> {
  ConvertGroupEvents(mlir::MLIRContext *context, HelperCache &helperCache)
      : mlir::OpConversionPattern<comp::GroupEvents>(context),
        helperCache(helperCache) {}

  mlir::LogicalResult
  matchAndRewrite(comp::GroupEvents op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const;

  HelperCache &helperCache;
};

template <typename SourceOp>
struct RemoveOpPattern : public mlir::OpConversionPattern<SourceOp> {
  using mlir::OpConversionPattern<SourceOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op.getOperation());
    return mlir::success();
  }
};

mlir::LogicalResult ConvertCreateExecEnv::matchAndRewrite(
    comp::CreateExecEnv op, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(
      op.getOperation(),
      mlir::ArrayRef<mlir::Type>{helperCache.llvmPointerType},
      rewriter.getSymbolRefAttr(kOclInit), mlir::ArrayRef<mlir::Value>{});
  std::cout << "Replace" << std::endl;
  return mlir::success();
}

mlir::LogicalResult ConvertDestroyExecEnv::matchAndRewrite(
    comp::DestroyExecEnv op, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(
      op.getOperation(), mlir::ArrayRef<mlir::Type>{},
      rewriter.getSymbolRefAttr(kOclDeinit),
      mlir::ArrayRef<mlir::Value>{operands[0]});
  return mlir::success();
}

mlir::LogicalResult
ConvertAlloc::matchAndRewrite(comp::Alloc op,
                              mlir::ArrayRef<mlir::Value> operands,
                              mlir::ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto &converter = *getTypeConverter();
  auto memRefType = op.getResult().getType().cast<mlir::MemRefType>();

  auto shape = memRefType.getShape();
  uint32_t numElement = 1;
  for (auto dim : shape) {
    numElement *= dim;
  }

  uint32_t elementTypeSize =
      llvm::divideCeil(memRefType.getElementTypeBitWidth(), 8);

  mlir::Value bufferByteSize = rewriter.create<LLVM::ConstantOp>(
      loc, helperCache.llvmInt32Type,
      rewriter.getI32IntegerAttr(numElement * elementTypeSize));

  mlir::Value hostPtr;
  if (auto hostMem = op.hostMem()) {
    mlir::LLVMTypeConverter llvmConverter(rewriter.getContext());
    auto hostType = hostMem.getType().cast<mlir::MemRefType>();
    // hostPtr = getTypeConverter()->materializeConversion(rewriter, loc,
    // getTypeConverter()->convertType(hostType), {hostMem});
    auto structMemRef = rewriter.create<LLVM::DialectCastOp>(
        loc, llvmConverter.convertType(hostType), hostMem);
    auto bufferDesc = mlir::MemRefDescriptor(structMemRef);
    auto flatMemAllocPtr = bufferDesc.allocatedPtr(rewriter, loc);
    hostPtr = rewriter.create<LLVM::BitcastOp>(
        loc, converter.convertType(hostType), flatMemAllocPtr);
  } else {
    hostPtr = rewriter.create<LLVM::NullOp>(loc, helperCache.llvmPointerType);
  }

  rewriter.replaceOpWithNewOp<LLVM::CallOp>(
      op.getOperation(),
      mlir::ArrayRef<mlir::Type>{helperCache.llvmPointerType},
      rewriter.getSymbolRefAttr(kOclAllocBuffer),
      mlir::ArrayRef<mlir::Value>{operands[0], bufferByteSize, hostPtr});

  return mlir::success();
}

mlir::LogicalResult ConvertDealloc::matchAndRewrite(
    comp::Dealloc op, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(
      op.getOperation(), mlir::ArrayRef<mlir::Type>{},
      rewriter.getSymbolRefAttr(kOclDeallocBuffer),
      mlir::ArrayRef<mlir::Value>(operands[0]));
  return mlir::success();
}

mlir::LogicalResult ConvertScheduleRead::matchAndRewrite(
    comp::ScheduleRead op, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  mlir::LLVMTypeConverter llvmConverter(rewriter.getContext());
  auto &converter = *getTypeConverter();
  auto hostMem = op.hostMem();
  auto hostType = hostMem.getType().cast<mlir::MemRefType>();
  // hostPtr = getTypeConverter()->materializeConversion(rewriter, loc,
  // getTypeConverter()->convertType(hostType), {hostMem});
  auto structMemRef = rewriter.create<LLVM::DialectCastOp>(
      loc, llvmConverter.convertType(hostType), hostMem);
  auto bufferDesc = mlir::MemRefDescriptor(structMemRef);
  auto flatMemAllocPtr = bufferDesc.allocatedPtr(rewriter, loc);
  auto hostPtr = rewriter.create<LLVM::BitcastOp>(
      loc, converter.convertType(hostType), flatMemAllocPtr);

  mlir::Value depPtr;
  if (auto depEvent = op.depEvent()) {
    depPtr = operands[3];
  } else {
    depPtr =
        rewriter.create<LLVM::NullOp>(op.getLoc(), helperCache.llvmPointerType);
  }
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(
      op.getOperation(),
      mlir::ArrayRef<mlir::Type>(helperCache.llvmPointerType),
      rewriter.getSymbolRefAttr(kOclEnqueueRead),
      mlir::ArrayRef<mlir::Value>{operands[2], operands[1], hostPtr, depPtr});
  return mlir::success();
}

mlir::LogicalResult
ConvertWait::matchAndRewrite(comp::Wait op,
                             mlir::ArrayRef<mlir::Value> operands,
                             mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(
      op.getOperation(), mlir::ArrayRef<mlir::Type>{},
      rewriter.getSymbolRefAttr(kOclWait),
      mlir::ArrayRef<mlir::Value>{operands[0]});
  return mlir::success();
}

mlir::LogicalResult ConvertScheduleFunc::matchAndRewrite(
    comp::ScheduleFunc op, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto execEnv = operands[0];
  auto launchOp = mlir::cast<gpu::LaunchFuncOp>(op.body().front().front());
  auto binaryName = static_cast<std::string>(launchOp.getKernelModuleName());
  if (kernelsMap.count(binaryName) == 0)
    return mlir::failure();
  auto &kernelInfo = kernelsMap[binaryName];
  // Get the pointer to the first character in the global string.
  mlir::Value globalPtr =
      rewriter.create<LLVM::AddressOfOp>(loc, kernelInfo.symbol);
  mlir::Value cst0 = rewriter.create<LLVM::ConstantOp>(
      loc, helperCache.llvmInt64Type,
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
  mlir::Value binaryPtr =
      rewriter.create<LLVM::GEPOp>(loc, helperCache.llvmPointerType, globalPtr,
                                   mlir::ArrayRef<mlir::Value>({cst0, cst0}));

  // mlir::Value binaryPtr = rewriter.create<LLVM::NullOp>(
  //   loc, helperCache.llvmPointerType);
  mlir::Value binarySize = rewriter.create<LLVM::ConstantOp>(
      loc, helperCache.llvmInt32Type,
      rewriter.getI32IntegerAttr(kernelInfo.bytes));
  auto createOp = rewriter.create<LLVM::CallOp>(
      loc, mlir::ArrayRef<mlir::Type>{helperCache.llvmPointerType},
      rewriter.getSymbolRefAttr(kOclCreateKernel),
      mlir::ArrayRef<mlir::Value>{execEnv, binaryPtr, binarySize});
  auto kernel = createOp.getResult(0);
  // // set args
  for (unsigned argI = 0; argI < launchOp.getNumKernelOperands(); ++argI) {
    mlir::Value argIndex = rewriter.create<LLVM::ConstantOp>(
        loc, helperCache.llvmInt32Type, rewriter.getI32IntegerAttr(argI));

    rewriter.create<LLVM::CallOp>(
        loc, mlir::ArrayRef<mlir::Type>{},
        rewriter.getSymbolRefAttr(kOclSetKernelArg),
        mlir::ArrayRef<mlir::Value>{execEnv, kernel, argIndex,
                                    launchOp.getKernelOperand(argI)});
  }

  auto gridSize = launchOp.getGridSizeOperandValues();
  auto blockSize = launchOp.getBlockSizeOperandValues();
  auto globalX = rewriter.create<mlir::MulIOp>(loc, gridSize.x, blockSize.x);
  auto globalY = rewriter.create<mlir::MulIOp>(loc, gridSize.y, blockSize.y);
  auto globalZ = rewriter.create<mlir::MulIOp>(loc, gridSize.z, blockSize.z);

  mlir::Value depPtr;
  if (auto depEvent = op.depEvent()) {
    depPtr = operands[1];
  } else {
    depPtr =
        rewriter.create<LLVM::NullOp>(op.getLoc(), helperCache.llvmPointerType);
  }

  rewriter.replaceOpWithNewOp<mlir::CallOp>(
      op.getOperation(),
      mlir::ArrayRef<mlir::Type>{helperCache.llvmPointerType},
      rewriter.getSymbolRefAttr(kOclEnqueueKernel),
      mlir::ArrayRef<mlir::Value>{execEnv, kernel, globalX, globalY, globalZ,
                                  blockSize.x, blockSize.y, blockSize.z,
                                  depPtr});
  // rewriter.replaceOpWithNewOp<LLVM::NullOp>(op.getOperation(),
  // helperCache.llvmPointerType);
  return mlir::success();
}

mlir::LogicalResult ConvertGroupEvents::matchAndRewrite(
    comp::GroupEvents op, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  std::vector<mlir::Value> callOperands;
  mlir::Value eventsCnt = rewriter.create<LLVM::ConstantOp>(
      op.getLoc(), helperCache.llvmInt32Type,
      rewriter.getI32IntegerAttr(operands.size()));

  callOperands.push_back(eventsCnt);

  for (auto &operand : operands) {
    callOperands.push_back(operand);
  }

  rewriter.replaceOpWithNewOp<LLVM::CallOp>(
      op.getOperation(),
      mlir::ArrayRef<mlir::Type>{helperCache.llvmPointerType},
      rewriter.getSymbolRefAttr(kOclGroupEvents),
      mlir::ArrayRef<mlir::Value>(callOperands));

  return mlir::success();
}

mlir::LogicalResult ConvertCompToOpenCl::convertCompOps() {
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalDialect<mlir::StandardOpsDialect>();
  target.addIllegalDialect<comp::COMPDialect>();

  CompLLVMTypeConverter typeConverter(helperCache);

  mlir::OwningRewritePatternList patterns;
  patterns.insert<ConvertCreateExecEnv, ConvertDestroyExecEnv, ConvertDealloc,
                  ConvertWait, ConvertGroupEvents>(&getContext(), helperCache);
  patterns.insert<ConvertAlloc, ConvertScheduleRead>(
      &getContext(), typeConverter, helperCache);
  patterns.insert<ConvertScheduleFunc>(&getContext(), helperCache, kernelsMap);
  patterns.insert<RemoveOpPattern<gpu::GPUModuleOp>,
                  RemoveOpPattern<spirv::ModuleOp>>(&getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                patterns, nullptr)))
    return mlir::failure();

  return mlir::success();
}

std::unique_ptr<mlir::Pass> createConvertCompToOpenClPass() {
  return std::make_unique<ConvertCompToOpenCl>();
}

} // namespace pmlc::conversion::comp
