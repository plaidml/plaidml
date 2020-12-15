// Copyright 2020, Intel Corporation

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

struct MemRefAccess {
  mlir::AffineValueMap accessMap;

  explicit MemRefAccess(PxaReadOpInterface op) {
    getAccessMap(op.getAffineMap(), op.getMapOperands(), &accessMap);
  }

  explicit MemRefAccess(PxaReduceOpInterface op) {
    getAccessMap(op.getAffineMap(), op.getMapOperands(), &accessMap);
  }

  void getAccessMap(mlir::AffineMap map,
                    mlir::SmallVector<mlir::Value, 8> operands,
                    mlir::AffineValueMap *accessMap);

  bool operator==(const MemRefAccess &rhs) const {
    mlir::AffineValueMap diff, lhsMap, rhsMap;
    mlir::AffineValueMap::difference(accessMap, rhs.accessMap, &diff);
    return llvm::all_of(diff.getAffineMap().getResults(),
                        [](mlir::AffineExpr expr) { return expr == 0; });
  }

  bool operator!=(const MemRefAccess &rhs) const { return !(*this == rhs); }
};

} // namespace pmlc::dialect::pxa
