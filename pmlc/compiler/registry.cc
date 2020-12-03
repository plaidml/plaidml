// Copyright 2019, Intel Corporation

#include "pmlc/compiler/registry.h"

#include <string>
#include <utility>

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

  void registerTarget(StringRef name, TargetPtr target) {
    if (targets.count(name)) {
      throw std::runtime_error(
          formatv("Target is already registered: {0}", name));
    }
    targets[name] = target;
  }

  TargetPtr resolve(StringRef name) {
    auto itTarget = targets.find(name);
    if (itTarget == targets.end()) {
      throw std::runtime_error(formatv("Could not find target: {0}", name));
    }
    return itTarget->second;
  }

  std::vector<StringRef> list() {
    auto keys = targets.keys();
    return std::vector<StringRef>(keys.begin(), keys.end());
  }

private:
  StringMap<TargetPtr> targets;
};

} // namespace

void registerTarget(StringRef name, TargetPtr target) {
  TargetRegistry::instance()->registerTarget(name, target);
}

TargetPtr resolveTarget(StringRef name) {
  return TargetRegistry::instance()->resolve(name);
}

std::vector<StringRef> listTargets() {
  return TargetRegistry::instance()->list();
}

} // namespace pmlc::compiler
