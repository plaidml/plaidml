// Copyright 2019, Intel Corporation

#include "pmlc/util/util.h"

#include "mlir/IR/Function.h"

using mlir::FuncOp;
using mlir::FunctionType;
using mlir::Operation;
using mlir::OperationName;
using mlir::SmallVector;
using mlir::Type;

namespace pmlc::util {

llvm::StringRef getOpName(const OperationName& name) {
  return name.getStringRef().drop_front(name.getDialect().size() + 1);
}

void UpdateFuncOpType(Operation* op) {
  if (auto funcOp = op->getParentOfType<FuncOp>()) {
    auto retOp = &funcOp.getOperation()->getRegion(0).front().back();
    auto funcType = funcOp.getType();
    if (funcType.getNumResults() == retOp->getNumOperands()) {
      SmallVector<Type, 4> retTypes(retOp->getOperandTypes());
      auto newType = FunctionType::get(funcType.getInputs(), retTypes, funcOp.getContext());
      if (funcType != newType) {
        funcOp.setType(newType);
      }
    }
  }
}

}  // namespace pmlc::util
