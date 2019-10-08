// Copyright 2019, Intel Corporation

#pragma once

#include "llvm/ADT/StringRef.h"

#include "mlir/IR/OperationSupport.h"

namespace pmlc {
namespace util {

llvm::StringRef getOpName(const mlir::OperationName& name);

}  // namespace util
}  // namespace pmlc
