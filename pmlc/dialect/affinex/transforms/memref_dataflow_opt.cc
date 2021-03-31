// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "pmlc/dialect/affinex/transforms/pass_detail.h"

using namespace mlir; // NOLINT

struct MemAccess {
  Value memref;
  AffineMap map;
  llvm::SmallVector<Value, 4> indices;
};

namespace llvm {
template <>
struct DenseMapInfo<MemAccess> {
  static inline MemAccess getEmptyKey() {
    MemAccess access;
    access.memref = DenseMapInfo<mlir::Value>::getEmptyKey();
    return access;
  }
  static inline MemAccess getTombstoneKey() {
    MemAccess access;
    access.memref = DenseMapInfo<mlir::Value>::getTombstoneKey();
    return access;
  }
  static unsigned getHashValue(const MemAccess &val) {
    return hash_combine(
        val.memref, val.map,
        hash_combine_range(val.indices.begin(), val.indices.end()));
  }
  static bool isEqual(const MemAccess &LHS, const MemAccess &RHS) {
    if (LHS.memref != RHS.memref) {
      return false;
    }

    if (LHS.map != RHS.map) {
      return false;
    }

    if (LHS.indices.size() != RHS.indices.size()) {
      return false;
    }

    for (size_t i = 0; i < LHS.indices.size(); ++i) {
      if (LHS.indices[i] != RHS.indices[i]) {
        return false;
      }
    }
    return true;
  }
};
} // namespace llvm

namespace pmlc::dialect::affinex {

struct AffinexMemRefDataFlowOpt
    : public AffinexMemRefDataFlowOptBase<AffinexMemRefDataFlowOpt> {

  void runOnFunction() override {
    Block *cur = nullptr;
    llvm::SmallVector<Operation *, 8> opsToErase;
    llvm::DenseMap<MemAccess, AffineWriteOpInterface> lastStoreOps;

    getFunction().walk([&](Operation *op) {
      if (op->getBlock() != cur) {
        lastStoreOps.clear();
        cur = op->getBlock();
      }

      if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op)) {
        MemAccess access{storeOp.getMemRef(), storeOp.getAffineMap(),
                         storeOp.getMapOperands()};
        // if we have already stored to this location in this block
        if (lastStoreOps.find(access) != lastStoreOps.end()) {
          auto lastStoreOp = lastStoreOps[access];
          // erase the previous store (later)
          opsToErase.push_back(lastStoreOp);
        }
        // update last store op
        lastStoreOps[access] = storeOp;
      } else if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) {
        MemAccess access{loadOp.getMemRef(), loadOp.getAffineMap(),
                         loadOp.getMapOperands()};
        // if we have already stored to this location in this block
        if (lastStoreOps.find(access) != lastStoreOps.end()) {
          auto lastStoreOp = lastStoreOps[access];
          // replace all uses of the load with the store value
          loadOp.getValue().replaceAllUsesWith(lastStoreOp.getValueToStore());
          // erase the load (later)
          opsToErase.push_back(loadOp);
        }
      }
    });

    // erase all redundant loads and stores
    for (auto *op : opsToErase) {
      op->erase();
    }
  }
};

std::unique_ptr<Pass> createAffinexMemRefDataFlowOpt() {
  return std::make_unique<AffinexMemRefDataFlowOpt>();
}
} // namespace pmlc::dialect::affinex
