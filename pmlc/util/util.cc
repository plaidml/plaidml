// Copyright 2019, Intel Corporation

#include "pmlc/util/util.h"

#include "mlir/IR/Function.h"

namespace pmlc::util {

llvm::StringRef getOpName(const mlir::OperationName& name) {
  return name.getStringRef().drop_front(name.getDialect().size() + 1);
}

void UpdateFuncOpType(mlir::Operation* op) {
  if (auto funcOp = op->getParentOfType<mlir::FuncOp>()) {
    auto retOp = &funcOp.getOperation()->getRegion(0).front().back();
    auto funcType = funcOp.getType();
    if (funcType.getNumResults() == retOp->getNumOperands()) {
      mlir::SmallVector<mlir::Type, 4> retTypes(retOp->getOperandTypes());
      auto newType = mlir::FunctionType::get(funcType.getInputs(), retTypes, funcOp.getContext());
      if (funcType != newType) {
        funcOp.setType(newType);
      }
    }
  }
}

}  // namespace pmlc::util
