#pragma once

#include "mlir/IR/Value.h"
#include "pmlc/util/enums.h"

namespace pmlc::util {

MemUse getMemoryUses(mlir::Value def);

inline bool doesRead(MemUse x) {
  return x == MemUse::read_only || x == MemUse::read_write;
}
inline bool doesWrite(MemUse x) {
  return x == MemUse::write_only || x == MemUse::read_write;
}

} // namespace pmlc::util
