// Copyright 2020, Intel Corporation

#pragma once

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace pmlc::rt {

struct SymbolRegistry {
  static SymbolRegistry *instance();
  void registerSymbol(llvm::StringRef symbol, void *ptr);
  void *resolve(llvm::StringRef symbol);

  llvm::StringMap<void *> symbols;
};

inline void registerSymbol(llvm::StringRef symbol, void *ptr) {
  SymbolRegistry::instance()->registerSymbol(symbol, ptr);
}

inline void *resolveSymbol(llvm::StringRef symbol) {
  return SymbolRegistry::instance()->resolve(symbol);
}

} // namespace pmlc::rt
