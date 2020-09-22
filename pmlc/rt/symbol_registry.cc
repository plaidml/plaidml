// Copyright 2020 Intel Corporation

#include "pmlc/rt/symbol_registry.h"

#include "pmlc/rt/internal.h"
#include "llvm/Support/FormatVariadic.h"

namespace pmlc::rt {

SymbolRegistry *SymbolRegistry::instance() {
  static SymbolRegistry registry;
  return &registry;
}

void SymbolRegistry::registerSymbol(llvm::StringRef symbol, void *ptr) {
  if (symbols.count(symbol)) {
    throw std::runtime_error(
        llvm::formatv("Symbol is already registered: {0}", symbol));
  }
  symbols[symbol] = ptr;
}

void *SymbolRegistry::resolve(llvm::StringRef symbol) {
  auto it = symbols.find(symbol);
  if (it == symbols.end()) {
    return nullptr;
  }
  return it->second;
}

} // namespace pmlc::rt
