// Copyright 2019, Intel Corporation

#pragma once

#include <functional>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace pmlc::compiler {

using TargetRegistryFunction = std::function<void(mlir::OpPassManager &)>;

void registerTarget(llvm::StringRef name,
                    const TargetRegistryFunction &function);

TargetRegistryFunction resolveTarget(llvm::StringRef name);

std::vector<llvm::StringRef> listTargets();

struct TargetRegistration {
  TargetRegistration(llvm::StringRef name, TargetRegistryFunction builder) {
    registerTarget(name, builder);
  }
};

struct SymbolRegistry {
  static SymbolRegistry *instance() {
    static SymbolRegistry registry;
    return &registry;
  }

  void registerSymbol(llvm::StringRef symbol, void *ptr) {
    if (symbols.count(symbol)) {
      throw std::runtime_error(
          formatv("Symbol is already registered: {0}", symbol));
    }
    symbols[symbol] = ptr;
  }

  void *resolve(llvm::StringRef symbol) {
    auto it = symbols.find(symbol);
    if (it == symbols.end()) {
      throw std::runtime_error(formatv("Could not find symbol: {0}", symbol));
    }
    return it->second;
  }

  llvm::StringMap<void *> symbols;
};

inline void registerSymbol(llvm::StringRef symbol, void *ptr) {
  SymbolRegistry::instance()->registerSymbol(symbol, ptr);
}

inline void *resolveSymbol(llvm::StringRef symbol) {
  return SymbolRegistry::instance()->resolve(symbol);
}

} // namespace pmlc::compiler
