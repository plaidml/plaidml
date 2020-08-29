// Copyright 2019, Intel Corporation

#include "pmlc/compiler/registry.h"

#include <string>

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

  void registerTarget(StringRef targetId, std::shared_ptr<Target> target) {
    auto [it, inserted] = registry.try_emplace(targetId, std::move(target));
    if (!inserted) {
      throw std::runtime_error(
          formatv("Target is already registered: {0}", targetId));
    }
  }

  std::shared_ptr<Target> resolve(StringRef targetId) {
    auto it = registry.find(targetId);
    if (it == registry.end()) {
      throw std::runtime_error(formatv("Could not find target: {0}", targetId));
    }
    return it->second;
  }

  std::vector<StringRef> list() {
    auto keys = registry.keys();
    return std::vector<StringRef>(keys.begin(), keys.end());
  }

private:
  StringMap<std::shared_ptr<Target>> registry;
};

} // namespace

void registerTarget(StringRef targetId, std::shared_ptr<Target> function) {
  TargetRegistry::instance()->registerTarget(targetId, std::move(function));
}

std::shared_ptr<Target> resolveTarget(StringRef targetId) {
  return TargetRegistry::instance()->resolve(targetId);
}

std::vector<StringRef> listTargets() {
  return TargetRegistry::instance()->list();
}

} // namespace pmlc::compiler
