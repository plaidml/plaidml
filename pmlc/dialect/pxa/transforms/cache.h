// Copyright 2020 Intel Corporation

#pragma once

#include "llvm/ADT/MapVector.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

struct CachePlan {
  template <typename OpType>
  struct OpInfo {
    OpType op;
    RelativeAccessPattern rap;
  };

  using LoadInfo = OpInfo<PxaLoadOp>;
  using ReduceInfo = OpInfo<PxaReduceOp>;

  struct Entry {
    RelativeAccessPattern rap;
    mlir::SmallVector<LoadInfo, 4> loads;
    mlir::SmallVector<ReduceInfo, 4> reduces;
    mlir::AffineParallelOp band;
    mlir::Value cache;
    bool copyInto = false;
    bool copyFrom = false;
  };

  llvm::MapVector<mlir::Value, Entry> entries;
  mlir::AffineParallelOp outerBand;
  mlir::AffineParallelOp middleBand;

  CachePlan(mlir::AffineParallelOp outerBand, mlir::AffineParallelOp middleBand)
      : outerBand(outerBand), middleBand(middleBand) {}

  void addLoad(PxaLoadOp op);

  void addReduce(PxaReduceOp op);

  void execute();
};

mlir::LogicalResult cacheLoad(mlir::AffineParallelOp par, PxaLoadOp load);

mlir::LogicalResult cacheReduce(mlir::AffineParallelOp par, PxaReduceOp reduce);

mlir::LogicalResult cacheLoadAsVector(mlir::AffineParallelOp par,
                                      PxaLoadOp load, int64_t reqVecSize = 0);

} // namespace pmlc::dialect::pxa
