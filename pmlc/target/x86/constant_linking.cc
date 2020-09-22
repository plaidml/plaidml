// Copyright 2020, Intel Corporation

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"

#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/target/x86/pass_detail.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace stdx = dialect::stdx;

namespace {

static constexpr const char *kGetConstantPtr = "plaidml_rt_get_constant_ptr";

struct ConstantMemRefOpConversion
    : public ConvertOpToLLVMPattern<stdx::ConstantMemRefOp> {
  using ConvertOpToLLVMPattern<stdx::ConstantMemRefOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto constOp = cast<stdx::ConstantMemRefOp>(op);
    MemRefType memRefType = constOp.getResult().getType().cast<MemRefType>();
    Location loc = op->getLoc();

    // Get actual sizes of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.  In case of
    // zero-dimensional memref, assume a scalar (size 1).
    SmallVector<Value, 4> sizes;
    getMemRefDescriptorSizes(loc, memRefType, operands, rewriter, sizes);

    auto ptrTy = getVoidPtrType();
    auto module = constOp.getParentOfType<ModuleOp>();

    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(kGetConstantPtr);
    auto funcSymbol = rewriter.getSymbolRefAttr(func);
    auto symbol = constOp.symbol().str();
    auto symbolValue = StringRef(symbol.c_str(), symbol.size() + 1);
    auto memRefSymbol =
        getGlobalString(loc, rewriter, symbol, symbolValue, module);
    auto allocatedBytePtr =
        rewriter
            .create<LLVM::CallOp>(loc, ptrTy, funcSymbol,
                                  ArrayRef<Value>{memRefSymbol})
            .getResult(0);
    auto elementPtrType = getElementPtrType(memRefType);
    auto allocatedTypePtr =
        rewriter.create<LLVM::BitcastOp>(loc, elementPtrType, allocatedBytePtr);

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(memRefType, strides, offset);
    (void)successStrides;
    assert(succeeded(successStrides) && "unexpected non-strided memref");
    assert(offset != MemRefType::getDynamicStrideOrOffset() &&
           "unexpected dynamic offset");

    // 0-D memref corner case: they have size 1.
    assert(
        ((memRefType.getRank() == 0 && strides.empty() && sizes.size() == 1) ||
         (strides.size() == sizes.size())) &&
        "unexpected number of strides");

    // Create the MemRef descriptor.
    auto memRefDescriptor = createMemRefDescriptor(
        loc, rewriter, memRefType, allocatedTypePtr, allocatedBytePtr,
        /*accessAlignment=*/nullptr, offset, strides, sizes);

    // Return the final value of the descriptor.
    rewriter.replaceOp(op, {memRefDescriptor});

    return success();
  }

private:
  // Returns bump = (alignment - (input % alignment))% alignment, which is the
  // increment necessary to align `input` to `alignment` boundary.
  // TODO: this can be made more efficient by just using a single addition
  // and two bit shifts: (ptr + align - 1)/align, align is always power of 2.
  Value createBumpToAlign(Location loc, OpBuilder &builder, Value input,
                          Value alignment) const {
    Value modAlign = builder.create<LLVM::URemOp>(loc, input, alignment);
    Value diff = builder.create<LLVM::SubOp>(loc, alignment, modAlign);
    Value shift = builder.create<LLVM::URemOp>(loc, diff, alignment);
    return shift;
  }

  /// Creates and populates the memref descriptor struct given all its fields.
  /// This method also performs any post allocation alignment needed for heap
  /// allocations when `accessAlignment` is non null. This is used with
  /// allocators that do not support alignment.
  MemRefDescriptor createMemRefDescriptor(
      Location loc, ConversionPatternRewriter &rewriter, MemRefType memRefType,
      Value allocatedTypePtr, Value allocatedBytePtr, Value accessAlignment,
      uint64_t offset, ArrayRef<int64_t> strides, ArrayRef<Value> sizes) const {
    auto elementPtrType = this->getElementPtrType(memRefType);
    auto structType = typeConverter.convertType(memRefType);
    auto memRefDescriptor = MemRefDescriptor::undef(rewriter, loc, structType);

    // Field 1: Allocated pointer, used for malloc/free.
    memRefDescriptor.setAllocatedPtr(rewriter, loc, allocatedTypePtr);

    // Field 2: Actual aligned pointer to payload.
    Value alignedBytePtr = allocatedTypePtr;
    if (accessAlignment) {
      // offset = (align - (ptr % align))% align
      Value intVal = rewriter.create<LLVM::PtrToIntOp>(
          loc, this->getIndexType(), allocatedBytePtr);
      Value offset = createBumpToAlign(loc, rewriter, intVal, accessAlignment);
      Value aligned = rewriter.create<LLVM::GEPOp>(
          loc, allocatedBytePtr.getType(), allocatedBytePtr, offset);
      alignedBytePtr = rewriter.create<LLVM::BitcastOp>(
          loc, elementPtrType, ArrayRef<Value>(aligned));
    }
    memRefDescriptor.setAlignedPtr(rewriter, loc, alignedBytePtr);

    // Field 3: Offset in aligned pointer.
    memRefDescriptor.setOffset(rewriter, loc,
                               createIndexConstant(rewriter, loc, offset));

    if (memRefType.getRank() == 0)
      // No size/stride descriptor in memref, return the descriptor value.
      return memRefDescriptor;

    // Fields 4 and 5: sizes and strides of the strided MemRef.
    // Store all sizes in the descriptor. Only dynamic sizes are passed in as
    // operands to AllocOp.
    Value runningStride = nullptr;
    // Iterate strides in reverse order, compute runningStride and strideValues.
    auto nStrides = strides.size();
    SmallVector<Value, 4> strideValues(nStrides, nullptr);
    for (unsigned i = 0; i < nStrides; ++i) {
      int64_t index = nStrides - 1 - i;
      if (strides[index] == MemRefType::getDynamicStrideOrOffset())
        // Identity layout map is enforced in the match function, so we compute:
        //   `runningStride *= sizes[index + 1]`
        runningStride = runningStride
                            ? rewriter.create<LLVM::MulOp>(loc, runningStride,
                                                           sizes[index + 1])
                            : createIndexConstant(rewriter, loc, 1);
      else
        runningStride = createIndexConstant(rewriter, loc, strides[index]);
      strideValues[index] = runningStride;
    }
    // Fill size and stride descriptors in memref.
    for (auto indexedSize : llvm::enumerate(sizes)) {
      int64_t index = indexedSize.index();
      memRefDescriptor.setSize(rewriter, loc, index, indexedSize.value());
      memRefDescriptor.setStride(rewriter, loc, index, strideValues[index]);
    }
    return memRefDescriptor;
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  Value getGlobalString(Location loc, ConversionPatternRewriter &rewriter,
                        StringRef name, StringRef value,
                        ModuleOp module) const {
    MLIRContext *context = rewriter.getContext();
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMType::getArrayTy(LLVM::LLVMType::getInt8Ty(context),
                                             value.size());
      global = rewriter.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                               LLVM::Linkage::Internal, name,
                                               rewriter.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
    Value zero = rewriter.create<LLVM::ConstantOp>(
        loc, LLVM::LLVMType::getInt64Ty(context),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    return rewriter.create<LLVM::GEPOp>(loc, getVoidPtrType(), globalPtr,
                                        ArrayRef<Value>({zero, zero}));
  }
};

} // namespace

void declareConstantLinkingFunctions(mlir::ModuleOp module) {
  MLIRContext *context = module.getContext();
  Location loc = module.getLoc();
  OpBuilder builder(module.getBodyRegion());

  auto ptrTy = LLVM::LLVMType::getInt8PtrTy(context);
  auto funcType =
      LLVM::LLVMType::getFunctionTy(ptrTy, ArrayRef<LLVM::LLVMType>{ptrTy},
                                    /*isVarArg=*/false);
  builder.create<LLVM::LLVMFuncOp>(loc, kGetConstantPtr, funcType);
}

void populateConstantLinkingPatterns(LLVMTypeConverter &converter,
                                     OwningRewritePatternList &patterns) {
  patterns.insert<ConstantMemRefOpConversion>(converter);
}

} // namespace pmlc::target::x86
