// Copyright 2020, Intel Corporation

#pragma once

// Copyright 2020 Intel Corporation
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "pmlc/util/tags.h"

namespace pmlc {

enum class MLFramework {
  Default, // No changes to layouts formats will be done
  NGraph
};

MLFramework getMLFramework(llvm::StringRef opName);

enum class TensorLayout { NXC, NCX, KCX };

TensorLayout getLayoutType(MLFramework framework, llvm::StringRef opName,
                           bool isConst = false);

inline llvm::StringRef layoutTag() { return "layout"; }

inline void setLayoutTag(mlir::Operation *op, TensorLayout layout) {
  setIntegerTag(op, layoutTag(), static_cast<int64_t>(layout));
}

inline bool hasLayoutTag(mlir::Operation *op) {
  return hasIntegerTag(op, layoutTag());
}

inline TensorLayout getLayoutTag(mlir::Operation *op) {
  return static_cast<TensorLayout>(getIntegerTag(op, layoutTag(), 0));
}

} // end namespace pmlc
