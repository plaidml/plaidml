// Copyright 2020, Intel Corporation

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"

namespace pmlc {

void copyTags(mlir::Operation *dst, mlir::Operation *src);

void clearTags(mlir::Operation *op);

// Set tags in op
void setTags(mlir::Operation *op, mlir::ArrayRef<mlir::StringRef> tags);

void clearTag(mlir::Operation *op, mlir::StringRef name);

void setUnitTag(mlir::Operation *op, mlir::StringRef name);

void setIntegerTag(mlir::Operation *op, mlir::StringRef name, int64_t val);

bool hasTags(mlir::Operation *op);

// Check if all tags exist in op and they are all true
bool hasAllTags(mlir::Operation *op, mlir::ArrayRef<mlir::StringRef> tags);

// Check if tag exists in op
bool hasTag(mlir::Operation *op, mlir::StringRef tag);

bool hasUnitTag(mlir::Operation *op, mlir::StringRef name);

bool hasIntegerTag(mlir::Operation *op, mlir::StringRef name);

int64_t getIntegerTag(mlir::Operation *op, mlir::StringRef name,
                      int64_t defaultVal);

void setIntegerArrayTag(mlir::Operation *op, mlir::StringRef name,
                        mlir::ArrayRef<int64_t> values);

bool getIntegerArrayTag(mlir::Operation *op, mlir::StringRef name,
                        mlir::SmallVectorImpl<int64_t> &out);

// List of 'known tags'.  These should probably be in their own headers.  But
// for now I just want to prevent a typo from compiling
inline mlir::StringRef subgroupSizeTag() { return "subgroupSize"; }
inline mlir::StringRef gpuThreadTag() { return "gpuThread"; }
inline mlir::StringRef gpuBlockTag() { return "gpuBlock"; }
inline mlir::StringRef cpuThreadTag() { return "cpuThread"; }

} // end namespace pmlc
