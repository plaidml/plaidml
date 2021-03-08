// Copyright 2020, Intel Corporation
#include "pmlc/conversion/comp_to_llvm/utils.h"

#include <string>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace pmlc::conversion::comp_to_llvm {

using namespace mlir; // NOLINT

LLVM::GlobalOp addGlobalString(OpBuilder &builder, Location loc,
                               StringRef symbol, StringRef string) {
  auto llvmStringType = LLVM::LLVMArrayType::get(
      IntegerType::get(builder.getContext(), 8), string.size());
  LLVM::GlobalOp globalOp = builder.create<LLVM::GlobalOp>(
      loc, llvmStringType,
      /*isConstant=*/true, LLVM::Linkage::Internal, symbol,
      builder.getStringAttr(string));
  return globalOp;
}

Value getPtrToGlobalString(OpBuilder &builder, Location &loc,
                           LLVM::GlobalOp globalOp) {
  auto llvmInt64Ty = IntegerType::get(builder.getContext(), 64);
  auto llvmPtrTy =
      LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8));
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, globalOp);
  Value cst0 = builder.create<LLVM::ConstantOp>(loc, llvmInt64Ty,
                                                builder.getI64IntegerAttr(0));
  Value stringPtr = builder.create<LLVM::GEPOp>(loc, llvmPtrTy, globalPtr,
                                                ArrayRef<Value>({cst0, cst0}));
  return stringPtr;
}

void getPtrToBinaryModule(OpBuilder &builder, Location &loc,
                          const BinaryModuleInfo &binaryInfo, Value &pointer,
                          Value &bytes) {
  pointer = getPtrToGlobalString(builder, loc, binaryInfo.symbol);

  auto llvmInt64Ty = IntegerType::get(builder.getContext(), 64);
  bytes = builder.create<LLVM::ConstantOp>(
      loc, llvmInt64Ty, builder.getI64IntegerAttr(binaryInfo.bytes));
}

} // namespace pmlc::conversion::comp_to_llvm
