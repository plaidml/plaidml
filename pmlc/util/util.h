// Copyright 2019, Intel Corporation

#pragma once

#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/StandardTypes.h"

namespace pmlc {
namespace util {

// Adjust the result types on the containing FuncOp if this op relates to an output
void UpdateFuncOpType(mlir::Operation* op);

llvm::StringRef getOpName(const mlir::OperationName& name);

template <typename Filter>
std::vector<mlir::AbstractOperation*> getAllOpsWith(mlir::MLIRContext* context, Filter filter) {
  std::vector<mlir::AbstractOperation*> ops;
  for (auto* op : context->getRegisteredOperations()) {
    if (filter(op)) {
      ops.emplace_back(op);
    }
  }
  return ops;
}

template <typename Interface>
std::vector<mlir::AbstractOperation*> getAllOpsWithInterface(mlir::MLIRContext* context) {
  return getAllOpsWith(context, [](mlir::AbstractOperation* op) { return op->getInterface<Interface>(); });
}

template <typename Set>
std::string getUniqueName(Set* names, llvm::StringRef name) {
  auto next = name.str();
  auto [it, isUnique] = names->insert(next);  // NOLINT(whitespace/braces)
  for (unsigned i = 0; !isUnique; i++) {
    next = llvm::formatv("{0}_{1}", name, i).str();
    std::tie(it, isUnique) = names->insert(next);
  }
  return next;
}

}  // namespace util
}  // namespace pmlc
