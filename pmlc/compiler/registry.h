// Copyright 2019, Intel Corporation

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

#include "pmlc/compiler/target.h"

namespace pmlc::compiler {

const llvm::StringMap<std::shared_ptr<Target>>& globalTargetRegistry();

void registerTarget(llvm::StringRef targetId, std::shared_ptr<Target> target);

struct TargetRegistration {
  TargetRegistration(llvm::StringRef targetId, std::shared_ptr<Target> target) {
    registerTarget(targetId, std::move(target));
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
          llvm::formatv("Symbol is already registered: {0}", symbol));
    }
    symbols[symbol] = ptr;
  }

  void *resolve(llvm::StringRef symbol) {
    auto it = symbols.find(symbol);
    if (it == symbols.end()) {
      return nullptr;
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
