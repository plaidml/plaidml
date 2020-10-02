// Copyright 2020, Intel Corporation
#include <string>

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/comp_to_llvm/pass_detail.h"
#include "pmlc/conversion/comp_to_llvm/passes.h"
#include "pmlc/conversion/comp_to_llvm/utils.h"
#include "pmlc/dialect/comp/ir/dialect.h"

namespace pmlc::conversion::comp_to_llvm {

namespace comp = pmlc::dialect::comp;
namespace spirv = mlir::spirv;
namespace LLVM = mlir::LLVM;
namespace gpu = mlir::gpu;

std::unique_ptr<BinaryModulesMap> getEmptyModulesMap() {
  return std::make_unique<BinaryModulesMap>();
}

static constexpr const char *kSpirvBinPrefix = "_pmlc_spirv_bin_";
static constexpr const char *kSpirvKernelPrefix = "_pmlc_spirv_kernel_";

mlir::LogicalResult serializeSpirvKernels(mlir::ModuleOp &op,
                                          BinaryModulesMap &map) {
  mlir::OpBuilder builder(op.getBodyRegion());
  std::vector<mlir::Operation *> toErase;

  mlir::WalkResult serializeWalk =
      op.walk([&](spirv::ModuleOp moduleOp) -> mlir::WalkResult {
        auto gpuModule = mlir::dyn_cast<gpu::GPUModuleOp>(
            moduleOp.getOperation()->getNextNode());
        if (!gpuModule)
          return mlir::WalkResult::interrupt();
        std::string gpuModuleName = gpuModule.getName().str();
        std::string binaryName = kSpirvBinPrefix + gpuModuleName;

        // Serialize spirv module.
        mlir::SmallVector<uint32_t, 0> moduleBinary;
        if (mlir::failed(spirv::serialize(moduleOp, moduleBinary)))
          return mlir::WalkResult::interrupt();

        LLVM::GlobalOp binaryOp =
            addGlobalString(builder, moduleOp.getLoc(), binaryName,
                            {reinterpret_cast<char *>(moduleBinary.data()),
                             moduleBinary.size() * 4});
        std::map<std::string, LLVM::GlobalOp> kernelNames;
        gpuModule.walk([&](gpu::GPUFuncOp funcOp) {
          if (!funcOp.isKernel())
            return;
          std::string kernelName = funcOp.getName().str();
          std::string symbol =
              kSpirvKernelPrefix + gpuModuleName + "_" + kernelName;
          // Make into null terminated string.
          auto nullTerminatedName = kernelName;
          nullTerminatedName.push_back('\0');
          LLVM::GlobalOp globalKernelName = addGlobalString(
              builder, funcOp.getLoc(), symbol, nullTerminatedName);
          kernelNames[kernelName] = globalKernelName;
        });
        // Update modules map.
        map[gpuModuleName] = {binaryOp, moduleBinary.size() * 4, kernelNames};

        // Add spirv and gpu modules to erase list.
        toErase.push_back(moduleOp.getOperation());
        toErase.push_back(gpuModule.getOperation());
        return mlir::WalkResult::advance();
      });
  if (serializeWalk.wasInterrupted())
    return mlir::failure();
  // Finally, erase processed operations.
  for (mlir::Operation *opToErase : toErase)
    opToErase->erase();
  return mlir::success();
}

static constexpr const char *kCastMemrefPrefix = "castMemrefToPtr";

static mlir::FailureOr<mlir::StringRef> getTypeMangling(mlir::Type type) {
  if (type.isInteger(8))
    return mlir::StringRef("I8");
  if (type.isInteger(16))
    return mlir::StringRef("I16");
  if (type.isInteger(32))
    return mlir::StringRef("I32");
  if (type.isInteger(64))
    return mlir::StringRef("I64");
  if (type.isBF16())
    return mlir::StringRef("BF16");
  if (type.isF16())
    return mlir::StringRef("F16");
  if (type.isF32())
    return mlir::StringRef("F32");
  if (type.isF64())
    return mlir::StringRef("F64");
  return mlir::failure();
}

static mlir::SmallVector<mlir::Type, 8>
getManglingTypes(mlir::MLIRContext *context) {
  return mlir::SmallVector<mlir::Type, 8>{
      mlir::IntegerType::get(8, context),  mlir::IntegerType::get(16, context),
      mlir::IntegerType::get(32, context), mlir::IntegerType::get(64, context),
      mlir::FloatType::getBF16(context),   mlir::FloatType::getF16(context),
      mlir::FloatType::getF32(context),    mlir::FloatType::getF64(context)};
}

void populateCommonPatterns(mlir::MLIRContext *context,
                            mlir::TypeConverter &typeConverter,
                            mlir::TypeConverter &signatureConverter,
                            mlir::OwningRewritePatternList &patterns) {
  // ==========================================================================
  // Type conversion patterns.
  // ==========================================================================
  LLVM::LLVMType llvmInt8Ptr = LLVM::LLVMType::getInt8PtrTy(context);
  typeConverter.addConversion(
      [=](comp::DeviceType deviceType) -> mlir::Optional<mlir::Type> {
        return llvmInt8Ptr;
      });
  // Identity conversion for LLVM types.
  typeConverter.addConversion([](LLVM::LLVMType type) { return type; });
  // Conversion between memref and int8 pointer.
  typeConverter.addConversion(
      [=](mlir::MemRefType type) { return llvmInt8Ptr; });
  // Noop materialization for identity conversion between LLVM types.
  typeConverter.addTargetMaterialization(
      [](mlir::OpBuilder &builder, LLVM::LLVMType type, mlir::ValueRange values,
         mlir::Location loc) -> mlir::Optional<mlir::Value> {
        if (values.size() != 1)
          return llvm::None;
        if (auto llvmSrcType = values[0].getType().dyn_cast<LLVM::LLVMType>()) {
          if (llvmSrcType == type)
            return values[0];
        }
        return llvm::None;
      });
  // Materialization for memref -> int8 pointer conversion.
  // TODO: This should materialize to dialect cast and pointer cast
  //       when support will be added to upstream.
  typeConverter.addTargetMaterialization(
      [=](mlir::OpBuilder &builder, LLVM::LLVMType type,
          mlir::ValueRange values,
          mlir::Location loc) -> mlir::Optional<mlir::Value> {
        if (type != llvmInt8Ptr || values.size() != 1)
          return llvm::None;
        mlir::MemRefType memRefType =
            values[0].getType().dyn_cast<mlir::MemRefType>();
        if (!memRefType)
          return llvm::None;

        mlir::Type unrankedMemRefType = mlir::UnrankedMemRefType::get(
            memRefType.getElementType(), /*memorySpace=*/0);
        mlir::Value unrankedBuffer = builder.create<mlir::MemRefCastOp>(
            loc, values[0], unrankedMemRefType);
        mlir::FailureOr<mlir::StringRef> typeManglingOr =
            getTypeMangling(memRefType.getElementType());
        if (mlir::failed(typeManglingOr))
          return llvm::None;
        mlir::StringRef typeMangling = typeManglingOr.getValue();

        mlir::Twine castFuncName = kCastMemrefPrefix + typeMangling;
        auto castOp = builder.create<mlir::CallOp>(
            loc, mlir::ArrayRef<mlir::Type>{llvmInt8Ptr},
            builder.getSymbolRefAttr(castFuncName.str()), unrankedBuffer);
        return castOp.getResult(0);
      });
  // ==========================================================================
  // Signature conversion patterns.
  // ==========================================================================
  signatureConverter.addConversion([](mlir::Type type) { return type; });
  signatureConverter.addConversion(
      [&](mlir::Type type) -> mlir::Optional<mlir::Type> {
        if (mlir::isa<comp::COMPDialect>(type.getDialect()))
          return typeConverter.convertType(type);
        return llvm::None;
      });
  // ==========================================================================
  // Operation conversion patterns.
  // ==========================================================================
  mlir::populateFuncOpTypeConversionPattern(patterns, context,
                                            signatureConverter);
}

void addCommonFunctionDeclarations(mlir::ModuleOp &module) {
  mlir::Location loc = module.getLoc();
  mlir::OpBuilder builder(module.getBody()->getTerminator());
  mlir::MLIRContext *context = builder.getContext();
  LLVM::LLVMType llvmInt8Ptr = LLVM::LLVMType::getInt8PtrTy(context);

  for (mlir::Type mangledType : getManglingTypes(context)) {
    mlir::FailureOr<mlir::StringRef> typeManglingOr =
        getTypeMangling(mangledType);
    mlir::StringRef typeMangling = typeManglingOr.getValue();
    mlir::Twine castFuncName = kCastMemrefPrefix + typeMangling;
    std::string castFuncNameStr = castFuncName.str();

    if (!module.lookupSymbol(castFuncNameStr)) {
      mlir::Type memRefType =
          mlir::UnrankedMemRefType::get(mangledType, /*memorySpace=*/0);
      builder.create<mlir::FuncOp>(
          loc, castFuncNameStr,
          mlir::FunctionType::get(mlir::ArrayRef<mlir::Type>{memRefType},
                                  {llvmInt8Ptr}, context));
    }
  }
}

} // namespace pmlc::conversion::comp_to_llvm
