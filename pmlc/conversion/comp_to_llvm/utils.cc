// Copyright 2020, Intel Corporation
#include "pmlc/conversion/comp_to_llvm/utils.h"

#include <string>

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"

namespace pmlc::conversion::comp_to_llvm {

namespace LLVM = mlir::LLVM;

LLVM::GlobalOp addGlobalString(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::StringRef symbol, mlir::StringRef string) {
  LLVM::LLVMType llvmInt8Type = LLVM::LLVMType::getInt8Ty(builder.getContext());
  LLVM::LLVMType llvmStringType =
      LLVM::LLVMType::getArrayTy(llvmInt8Type, string.size());

  LLVM::GlobalOp globalOp = builder.create<LLVM::GlobalOp>(
      loc, llvmStringType,
      /*isConstant=*/true, LLVM::Linkage::Internal, symbol,
      builder.getStringAttr(string));
  return globalOp;
}

mlir::Value getPtrToGlobalString(mlir::OpBuilder &builder, mlir::Location &loc,
                                 mlir::LLVM::GlobalOp globalOp) {
  LLVM::LLVMType llvmInt32Ty = LLVM::LLVMType::getInt32Ty(builder.getContext());
  LLVM::LLVMType llvmPtrTy = LLVM::LLVMType::getInt8PtrTy(builder.getContext());
  mlir::Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, globalOp);
  mlir::Value cst0 = builder.create<LLVM::ConstantOp>(
      loc, llvmInt32Ty, builder.getI32IntegerAttr(0));
  mlir::Value stringPtr = builder.create<LLVM::GEPOp>(
      loc, llvmPtrTy, globalPtr, mlir::ArrayRef<mlir::Value>({cst0, cst0}));
  return stringPtr;
}

void getPtrToBinaryModule(mlir::OpBuilder &builder, mlir::Location &loc,
                          const BinaryModuleInfo &binaryInfo,
                          mlir::Value &pointer, mlir::Value &bytes) {
  pointer = getPtrToGlobalString(builder, loc, binaryInfo.symbol);

  LLVM::LLVMType llvmInt32Ty = LLVM::LLVMType::getInt32Ty(builder.getContext());
  bytes = builder.create<LLVM::ConstantOp>(
      loc, llvmInt32Ty, builder.getI32IntegerAttr(binaryInfo.bytes));
}

mlir::Value deviceMemrefToMem(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value memref) {
  // Start with a valid buffer.
  mlir::Value buffer =
      mlir::MemRefDescriptor{memref}.allocatedPtr(builder, loc);
  while (memref) {
    // Attempt to trace the memref to a source buffer.
    auto prevBuffer = buffer;
    do {
      auto srcOp = memref.getDefiningOp();
      if (!srcOp) {
        break;
      }
      memref = llvm::TypeSwitch<mlir::Operation *, mlir::Value>(srcOp)
                   .Case([&](LLVM::InsertValueOp ivOp) {
                     auto pos = ivOp.position().getValue();
                     if (pos.size() == 1) {
                       auto intAttr = pos.front().dyn_cast<mlir::IntegerAttr>();
                       if (intAttr && intAttr.getInt() == 0) {
                         buffer = ivOp.value();
                         return mlir::Value{};
                       }
                     }
                     return ivOp.container();
                   })
                   .Default([](mlir::Operation * /* op */) {
                     return mlir::Value{};
                   });
    } while (memref);

    if (prevBuffer == buffer) {
      // We weren't able to trace the defining memref back to an earlier buffer.
      break;
    }

    // Attempt to trace the buffer to a source memref.
    for (;;) {
      auto srcOp = buffer.getDefiningOp();
      if (!srcOp) {
        break;
      }
      auto srcBuffer =
          llvm::TypeSwitch<mlir::Operation *, mlir::Value>(srcOp)
              .Case([](LLVM::AddrSpaceCastOp castOp) { return castOp.arg(); })
              .Case([](LLVM::BitcastOp castOp) { return castOp.arg(); })
              .Case([&](LLVM::ExtractValueOp evOp) {
                auto pos = evOp.position().getValue();
                if (pos.size() == 1) {
                  auto intAttr = pos.front().dyn_cast<mlir::IntegerAttr>();
                  if (intAttr && intAttr.getInt() == 0) {
                    memref = evOp.container();
                  }
                }
                return mlir::Value{};
              })
              .Default(
                  [](mlir::Operation * /* op */) { return mlir::Value{}; });
      if (!srcBuffer) {
        break;
      }
      buffer = srcBuffer;
    }
  }

  // The buffer is now an LLVM pointer-to-element-type in some address space.

  // If the address space is not the default (zero) address space, cast it.
  auto bufferTy = buffer.getType().cast<LLVM::LLVMPointerType>();
  if (bufferTy.getAddressSpace() != 0) {
    buffer = builder.create<LLVM::AddrSpaceCastOp>(
        loc, bufferTy.getElementType().getPointerTo(), buffer);
  }

  // If the element type is not i8, cast it.
  if (!bufferTy.getElementType().isIntegerTy(8)) {
    buffer = builder.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMType::getInt8PtrTy(builder.getContext()), buffer);
  }

  return buffer;
}

mlir::Value hostMemrefToMem(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value memref) {
  auto buffer = mlir::MemRefDescriptor{memref}.allocatedPtr(builder, loc);

  // The buffer is now an LLVM pointer-to-element-type in the host address
  // space.
  auto bufferTy = buffer.getType().cast<LLVM::LLVMPointerType>();

  // If the element type is not i8, cast it.
  if (!bufferTy.getElementType().isIntegerTy(8)) {
    buffer = builder.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMType::getInt8PtrTy(builder.getContext()), buffer);
  }

  return buffer;
}

mlir::Value indexToInt(mlir::OpBuilder &builder, mlir::Location &loc,
                       mlir::LLVMTypeConverter &typeConverter,
                       mlir::Value value) {
  if (value.getType().isIndex()) {
    value = builder.create<mlir::IndexCastOp>(
        loc, builder.getIntegerType(typeConverter.getIndexTypeBitwidth()),
        value);
  }
  return typeConverter.materializeTargetConversion(
      builder, loc, typeConverter.getIndexType(), {value});
}

} // namespace pmlc::conversion::comp_to_llvm
