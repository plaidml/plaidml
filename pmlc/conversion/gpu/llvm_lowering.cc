#include <functional>

#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::conversion::gpu {
#define PASS_NAME "pmlc-convert-std-to-llvm"

namespace {
// Extract an LLVM IR type from the LLVM IR dialect type.
static LLVM::LLVMType unwrap(Type type) {
  if (!type)
    return nullptr;
  auto *mlirContext = type.getContext();
  auto wrappedLLVMType = type.dyn_cast<LLVM::LLVMType>();
  if (!wrappedLLVMType)
    emitError(UnknownLoc::get(mlirContext),
              "conversion resulted in a non-LLVM type");
  return wrappedLLVMType;
}

/// Convert a MemRef type to a bare pointer to the MemRef element type.
static Type convertMemRefTypeToBarePtr(LLVMTypeConverter &converter,
                                       MemRefType type) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(type, strides, offset)))
    return {};

  LLVM::LLVMType elementType =
      unwrap(converter.convertType(type.getElementType()));
  if (!elementType)
    return {};
  return elementType.getPointerTo(type.getMemorySpace());
}

/// Callback to convert function argument types. It converts MemRef function
/// arguments to bare pointers to the MemRef element type.
LogicalResult barePtrFuncArgTypeConverter(LLVMTypeConverter &converter,
                                          Type type,
                                          SmallVectorImpl<Type> &result) {
  if (auto memrefTy = type.dyn_cast<MemRefType>()) {
    auto llvmTy = convertMemRefTypeToBarePtr(converter, memrefTy);
    if (!llvmTy)
      return failure();

    result.push_back(llvmTy);
    return success();
  }

  if (type.isa<UnrankedMemRefType>()) {
    mlir::Type llvmInt64Type =
        LLVM::LLVMType::getInt64Ty(converter.getDialect());
    mlir::Type llvmPointerType =
        LLVM::LLVMType::getInt8PtrTy(converter.getDialect());
    result.append({llvmInt64Type, llvmPointerType});
    return success();
  }

  auto llvmTy = converter.convertType(type);
  if (!llvmTy)
    return failure();

  result.push_back(llvmTy);
  return success();
}

/// A pass converting MLIR operations into the LLVM IR dialect.
struct LLVMLoweringPass : public ModulePass<LLVMLoweringPass> {
  /// Creates an LLVM lowering pass.
  explicit LLVMLoweringPass(bool useAlloca, bool useBarePtrCallConv,
                            bool emitCWrappers) {
    this->useAlloca = useAlloca;
    this->useBarePtrCallConv = useBarePtrCallConv;
    this->emitCWrappers = emitCWrappers;
  }
  LLVMLoweringPass() {}
  LLVMLoweringPass(const LLVMLoweringPass &pass) {}

  /// Run the dialect converter on the module.
  void runOnModule() override {
    if (useBarePtrCallConv && emitCWrappers) {
      getModule().emitError()
          << "incompatible conversion options: bare-pointer calling convention "
             "and C wrapper emission";
      signalPassFailure();
      return;
    }

    ModuleOp m = getModule();

    LLVMTypeConverterCustomization customs;
    customs.funcArgConverter = useBarePtrCallConv ? barePtrFuncArgTypeConverter
                                                  : structFuncArgTypeConverter;
    LLVMTypeConverter typeConverter(&getContext(), customs);

    OwningRewritePatternList patterns;
    if (useBarePtrCallConv)
      populateStdToLLVMBarePtrConversionPatterns(typeConverter, patterns,
                                                 useAlloca);
    else
      populateStdToLLVMConversionPatterns(typeConverter, patterns, useAlloca,
                                          emitCWrappers);

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(m, target, patterns, &typeConverter)))
      signalPassFailure();
  }

  /// Use `alloca` instead of `call @malloc` for converting std.alloc.
  Option<bool> useAlloca{
      *this, "use-alloca",
      llvm::cl::desc("Replace emission of malloc/free by alloca"),
      llvm::cl::init(false)};

  /// Convert memrefs to bare pointers in function signatures.
  Option<bool> useBarePtrCallConv{
      *this, "use-bare-ptr-memref-call-conv",
      llvm::cl::desc("Replace FuncOp's MemRef arguments with "
                     "bare pointers to the MemRef element types"),
      llvm::cl::init(false)};

  /// Emit wrappers for C-compatible pointer-to-struct memref descriptors.
  Option<bool> emitCWrappers{
      *this, "emit-c-wrappers",
      llvm::cl::desc("Emit C-compatible wrapper functions"),
      llvm::cl::init(false)};
};
} // end namespace

std::unique_ptr<mlir::Pass> createLowerToLLVMPass(bool useAlloca,
                                                  bool useBarePtrCallConv,
                                                  bool emitCWrappers) {
  return std::make_unique<LLVMLoweringPass>(useAlloca, useBarePtrCallConv,
                                            emitCWrappers);
}

static PassRegistration<LLVMLoweringPass>
    pass(PASS_NAME, "Convert scalar and vector operations from the "
                    "Standard to the LLVM dialect");
} // namespace pmlc::conversion::gpu
