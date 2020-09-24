// Copyright 2020, Intel Corporation

#include "pmlc/rt/runtime_registry.h"

#include <list>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "pmlc/util/logging.h"
#include "llvm/Support/FormatVariadic.h"

namespace pmlc::rt {
namespace {

class RuntimeRegistry {
public:
  static RuntimeRegistry *instance() {
    static RuntimeRegistry reg;
    return &reg;
  }

  void registerFactory(mlir::StringRef id, Factory factory) {
    factories.emplace_back(id, std::move(factory));
  }

  void registerRuntime(mlir::StringRef id, std::shared_ptr<Runtime> runtime) {
    auto inserted = runtimes.emplace(id, std::move(runtime)).second;
    if (!inserted) {
      throw std::runtime_error{
          llvm::formatv("Multiple runtime implementations found for {0}", id)};
    }
  }

  Runtime *getRuntime(mlir::StringRef id) {
    auto it = runtimes.find(id.str());
    if (it == runtimes.end()) {
      throw std::runtime_error{llvm::formatv("Unable to find {0} runtime", id)};
    }
    return it->second.get();
  }

  const std::unordered_map<std::string, std::shared_ptr<Runtime>> &
  getRuntimeMap() {
    return runtimes;
  }

  void initRuntimes() {
    while (!factories.empty()) {
      auto [id, factory] = factories.front();
      factories.pop_front();
      std::shared_ptr<Runtime> runtime;
      try {
        runtime = factory();
      } catch (...) {
        continue;
      }
      registerRuntime(id, std::move(runtime));
    }
  }

private:
  std::list<std::pair<std::string, Factory>> factories;
  std::unordered_map<std::string, std::shared_ptr<Runtime>> runtimes;
};

} // namespace

void registerFactory(mlir::StringRef id, Factory factory) {
  RuntimeRegistry::instance()->registerFactory(id, std::move(factory));
}

Runtime *getRuntime(mlir::StringRef id) {
  return RuntimeRegistry::instance()->getRuntime(id);
}

const std::unordered_map<std::string, std::shared_ptr<Runtime>> &
getRuntimeMap() {
  return RuntimeRegistry::instance()->getRuntimeMap();
}

void registerRuntime(mlir::StringRef id, std::shared_ptr<Runtime> runtime) {
  RuntimeRegistry::instance()->registerRuntime(id, std::move(runtime));
}

void initRuntimes() { RuntimeRegistry::instance()->initRuntimes(); }

} // namespace pmlc::rt
