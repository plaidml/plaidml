// Copyright 2019, Intel Corporation

#pragma once

#include <functional>
#include <vector>

#include "llvm/ADT/StringRef.h"

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

void registerSymbol(llvm::StringRef symbol, void *ptr);

void *resolveSymbol(llvm::StringRef symbol);

struct SymbolRegistration {
  SymbolRegistration(llvm::StringRef symbol, void *ptr) {
    registerSymbol(symbol, ptr);
  }
};

} // namespace pmlc::compiler
