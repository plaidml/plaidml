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

  void registerTarget(StringRef name, TargetFactory factory) {
    if (factories.count(name)) {
      throw std::runtime_error(
          formatv("Target is already registered: {0}", name));
    }
    factories[name] = factory;
  }

  TargetPtr resolve(StringRef name) {
    auto itTarget = targets.find(name);
    if (itTarget == targets.end()) {
      auto itFactory = factories.find(name);
      if (itFactory == factories.end()) {
        throw std::runtime_error(formatv("Could not find target: {0}", name));
      }
      TargetFactory factory = itFactory->second;
      TargetPtr target = factory();
      std::tie(itTarget, std::ignore) =
          targets.insert(std::make_pair(name, target));
    }
    return itTarget->second;
  }

  std::vector<StringRef> list() {
    auto keys = factories.keys();
    return std::vector<StringRef>(keys.begin(), keys.end());
  }

private:
  StringMap<TargetFactory> factories;
  StringMap<TargetPtr> targets;
};

} // namespace

void registerTarget(StringRef name, TargetFactory factory) {
  TargetRegistry::instance()->registerTarget(name, factory);
}

TargetPtr resolveTarget(StringRef name) {
  return TargetRegistry::instance()->resolve(name);
}

std::vector<StringRef> listTargets() {
  return TargetRegistry::instance()->list();
}

} // namespace pmlc::compiler
