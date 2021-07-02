// Copyright 2020, Intel Corporation

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"

namespace pmlc {

void clearTag(mlir::Operation *op, llvm::StringRef name);

void clearTags(mlir::Operation *op);

void copyTags(mlir::Operation *dst, mlir::Operation *src);

int64_t getIntegerTag(mlir::Operation *op, llvm::StringRef name,
                      int64_t defaultVal);

// Check if tag exists in op
bool hasTag(mlir::Operation *op, llvm::StringRef tag);

bool hasTags(mlir::Operation *op);

// Check if all tags exist in op and they are all true
bool hasAllTags(mlir::Operation *op, llvm::ArrayRef<llvm::StringRef> tags);

bool hasIntegerTag(mlir::Operation *op, llvm::StringRef name);

bool hasUnitTag(mlir::Operation *op, llvm::StringRef name);

// Set tags in op
void setTags(mlir::Operation *op, llvm::ArrayRef<llvm::StringRef> tags);

void setIntegerTag(mlir::Operation *op, llvm::StringRef name, int64_t val);

void setUnitTag(mlir::Operation *op, llvm::StringRef name);

} // end namespace pmlc
