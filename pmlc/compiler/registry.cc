// Copyright 2019, Intel Corporation

#include "pmlc/compiler/registry.h"

using namespace llvm; // NOLINT[build/namespaces]
using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::compiler {

namespace {

llvm::StringMap<std::shared_ptr<Target>>& getInstance() {
  static llvm::StringMap<std::shared_ptr<Target>> instance;
  return instance;
}

} // namespace

void registerTarget(StringRef targetId, std::shared_ptr<Target> target) {
  auto [it, inserted] = getInstance().try_emplace(targetId, std::move(target));
  if (!inserted) {
    throw std::runtime_error(formatv("Target is already registered: {0}", targetId));
  }
}

const llvm::StringMap<std::shared_ptr<Target>>& globalTargetRegistry() {
  return getInstance();
}

} // namespace pmlc::compiler
