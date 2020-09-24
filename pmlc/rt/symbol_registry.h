// Copyright 2020, Intel Corporation

#pragma once

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace pmlc::rt {

struct SymbolRegistry {
  static SymbolRegistry *instance();
  void registerSymbol(mlir::StringRef symbol, void *ptr);
  void *resolve(mlir::StringRef symbol);

  ::llvm::StringMap<void *> symbols;
};

inline void registerSymbol(mlir::StringRef symbol, void *ptr) {
  SymbolRegistry::instance()->registerSymbol(symbol, ptr);
}

inline void *resolveSymbol(mlir::StringRef symbol) {
  return SymbolRegistry::instance()->resolve(symbol);
}

} // namespace pmlc::rt
