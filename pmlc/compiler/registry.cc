// Copyright 2019, Intel Corporation

#include "pmlc/compiler/registry.h"

#include <string>
#include <unordered_map>

#include "llvm/Support/FormatVariadic.h"

#include "mlir/Support/LLVM.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::compiler {

namespace {

class TargetRegistry {
public:
  static TargetRegistry *Instance() {
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
    std::vector<StringRef> ret;
    ret.reserve(registry.size());
    for (const auto &[key, _] : registry) {
      ret.emplace_back(key);
    }
    return ret;
  }

private:
  std::unordered_map<std::string, TargetRegistryFunction> registry;
};

} // namespace

void registerTarget(mlir::StringRef name,
                    const TargetRegistryFunction &function) {
  TargetRegistry::Instance()->registerTarget(name, function);
}

TargetRegistryFunction resolveTarget(mlir::StringRef name) {
  return TargetRegistry::Instance()->resolve(name);
}

std::vector<mlir::StringRef> listTargets() {
  return TargetRegistry::Instance()->list();
}

} // namespace pmlc::compiler
