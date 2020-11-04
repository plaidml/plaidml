// Copyright 2020 Intel Corporation
#pragma once

#include "mlir/Support/LLVM.h"

#include "pmlc/dialect/comp/ir/dialect.h"

namespace pmlc::dialect::comp {

/// Class tracking status of synchronization between memories.
/// Two memories are in sync when they hold the same content.
class MemorySynchronizationTracker {
public:
  /// Returns whether two memories hold the same content.
  bool areInSync(mlir::Value a, mlir::Value b);
  /// Returns whether specified memory is tracked.
  bool isTracked(mlir::Value mem);

  /// Modifies synchronization status by effects of "operation".
  /// Returns true if tracked connections changed.
  bool handleOperation(mlir::Operation *operation);

  bool handleTransferOp(MemoryTransferOpInterface op);
  bool handleAllocOp(Alloc op);
  bool handleGeneralOp(mlir::Operation *op);

  bool syncMemory(mlir::Value src, mlir::Value dst);
  bool desyncMemory(mlir::Value mem);

private:
  mlir::DenseMap<mlir::Value, mlir::DenseSet<mlir::Value>> inSyncMap;
};

} // namespace pmlc::dialect::comp
