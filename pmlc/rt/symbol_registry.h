// Copyright 2020, Intel Corporation

#pragma once

#include "pmlc/rt/registry.h"

namespace pmlc::rt {

using SymbolRegistry = Registry<void *>;

inline void registerSymbol(llvm::StringRef symbol, void *ptr) {
  SymbolRegistry::instance()->registerItem(symbol, ptr);
}

inline void *resolveSymbol(llvm::StringRef symbol) {
  return SymbolRegistry::instance()->resolve(symbol);
}

} // namespace pmlc::rt
