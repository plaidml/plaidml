// Copyright 2020, Intel Corporation

#pragma once

#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpDefinition.h"

namespace pmlc::dialect::pxa {

struct PxaReadMemAccess {
  mlir::Value source;
  mlir::Value target;
  mlir::AffineValueMap valueMap;

  mlir::MemRefType getMemRefType() {
    return source.getType().cast<mlir::MemRefType>();
  }
};

struct PxaWriteMemAccess {
  mlir::Value source;
  mlir::Value target;
  mlir::Value result;
  mlir::AffineValueMap valueMap;
  mlir::AtomicRMWKind reduction;

  mlir::MemRefType getMemRefType() {
    return target.getType().cast<mlir::MemRefType>();
  }
};

} // namespace pmlc::dialect::pxa

#include "pmlc/dialect/pxa/ir/interfaces.h.inc"
