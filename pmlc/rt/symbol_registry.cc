// Copyright 2020 Intel Corporation

#include "pmlc/rt/symbol_registry.h"

#include "pmlc/rt/internal.h"
#include "llvm/Support/FormatVariadic.h"

namespace pmlc::rt {

llvm::StringMap<void *> &getSymbolMap() {
  static llvm::StringMap<void *> symbolMap;
  return symbolMap;
}

void registerSymbol(llvm::StringRef symbol, void *ptr) {
  auto &symbols = getSymbolMap();
  if (symbols.count(symbol)) {
    throw std::runtime_error(
        llvm::formatv("Symbol is already registered: {0}", symbol));
  }
  symbols[symbol] = ptr;
}

void *resolveSymbol(llvm::StringRef symbol) {
  auto symbols = getSymbolMap();
  auto it = symbols.find(symbol);
  if (it == symbols.end()) {
    return nullptr;
  }
  return it->second;
}

} // namespace pmlc::rt
