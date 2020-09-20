// Copyright 2019, Intel Corporation

#pragma once

#include <functional>
#include <string>
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

std::vector<std::string> listTargets();

struct TargetRegistration {
  TargetRegistration(llvm::StringRef name, TargetRegistryFunction builder) {
    registerTarget(name, builder);
  }
};

} // namespace pmlc::compiler
