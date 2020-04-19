// Copyright 2019, Intel Corporation

#include "pmlc/compiler/registry.h"

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/Support/LLVM.h"

using namespace llvm; // NOLINT[build/namespaces]
using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::compiler {

namespace {

class TargetRegistry {
public:
  static TargetRegistry *instance() {
    static TargetRegistry registry;
    return &registry;
  }

  void registerTarget(StringRef name, const TargetRegistryFunction &function) {
    if (registry.count(name)) {
      throw std::runtime_error(
          formatv("Target is already registered: {0}", name));
    }
    registry[name] = function;
  }

  TargetRegistryFunction resolve(StringRef name) {
    auto it = registry.find(name);
    if (it == registry.end()) {
      throw std::runtime_error(formatv("Could not find target: {0}", name));
    }
    return it->second;
  }

  std::vector<StringRef> list() {
    auto keys = registry.keys();
    return std::vector<StringRef>(keys.begin(), keys.end());
  }

private:
  StringMap<TargetRegistryFunction> registry;
};

class SymbolRegistry {
public:
  static SymbolRegistry *instance() {
    static SymbolRegistry registry;
    return &registry;
  }

  void registerSymbol(StringRef symbol, void *ptr) {
    if (registry.count(symbol)) {
      throw std::runtime_error(
          formatv("Symbol is already registered: {0}", symbol));
    }
    registry[symbol] = ptr;
  }

  void *resolve(StringRef symbol) {
    auto it = registry.find(symbol);
    if (it == registry.end()) {
      throw std::runtime_error(formatv("Could not find symbol: {0}", symbol));
    }
    return it->second;
  }

private:
  StringMap<void *> registry;
};

} // namespace

void registerTarget(StringRef name, const TargetRegistryFunction &function) {
  TargetRegistry::instance()->registerTarget(name, function);
}

TargetRegistryFunction resolveTarget(StringRef name) {
  return TargetRegistry::instance()->resolve(name);
}

std::vector<StringRef> listTargets() {
  return TargetRegistry::instance()->list();
}

void registerSymbol(llvm::StringRef symbol, void *ptr) {
  SymbolRegistry::instance()->registerSymbol(symbol, ptr);
}

void *resolveSymbol(llvm::StringRef symbol) {
  return SymbolRegistry::instance()->resolve(symbol);
}

} // namespace pmlc::compiler
