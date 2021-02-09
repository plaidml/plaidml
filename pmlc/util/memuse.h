#pragma once

#include "mlir/IR/Value.h"
#include "pmlc/util/enums.h"

namespace pmlc::util {

MemUse getMemoryUses(mlir::Value def);

// Checks if 'maybeAlloc' is allocation like, and if so, find it's assocaited
// deallocation.  Otherwise, return nullptr;
mlir::Operation *findDeallocPair(mlir::Operation *maybeAlloc);

inline bool doesRead(MemUse x) {
  return x == MemUse::read_only || x == MemUse::read_write;
}
inline bool doesWrite(MemUse x) {
  return x == MemUse::write_only || x == MemUse::read_write;
}

} // namespace pmlc::util
