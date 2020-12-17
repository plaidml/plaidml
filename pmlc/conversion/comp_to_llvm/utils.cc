// Copyright 2020, Intel Corporation
#include "pmlc/conversion/comp_to_llvm/utils.h"

#include <string>

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
  LLVM::LLVMType llvmInt64Ty = LLVM::LLVMType::getInt64Ty(builder.getContext());
  LLVM::LLVMType llvmPtrTy = LLVM::LLVMType::getInt8PtrTy(builder.getContext());
  mlir::Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, globalOp);
  mlir::Value cst0 = builder.create<LLVM::ConstantOp>(
      loc, llvmInt64Ty, builder.getI64IntegerAttr(0));
  mlir::Value stringPtr = builder.create<LLVM::GEPOp>(
      loc, llvmPtrTy, globalPtr, mlir::ArrayRef<mlir::Value>({cst0, cst0}));
  return stringPtr;
}

void getPtrToBinaryModule(mlir::OpBuilder &builder, mlir::Location &loc,
                          const BinaryModuleInfo &binaryInfo,
                          mlir::Value &pointer, mlir::Value &bytes) {
  pointer = getPtrToGlobalString(builder, loc, binaryInfo.symbol);

  LLVM::LLVMType llvmInt64Ty = LLVM::LLVMType::getInt64Ty(builder.getContext());
  bytes = builder.create<LLVM::ConstantOp>(
      loc, llvmInt64Ty, builder.getI64IntegerAttr(binaryInfo.bytes));
}

} // namespace pmlc::conversion::comp_to_llvm
