// Copyright 2020, Intel Corporation

#include "pmlc/rt/runtime_registry.h"

#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "pmlc/util/logging.h"
#include "llvm/Support/FormatVariadic.h"

namespace pmlc::rt {
namespace {

class DuplicateRuntimeError final : public std::runtime_error {
public:
  explicit DuplicateRuntimeError(llvm::StringRef id)
      : std::runtime_error{llvm::formatv(
            "Multiple runtime implementations found for {0}", id)} {}
};

class Memoizer final {
public:
  explicit Memoizer(Factory factory) : factory{std::move(factory)} {}
  Runtime *get() {
    std::call_once(init_once, [this]() { runtime = factory(); });
    return runtime.get();
  }

private:
  std::once_flag init_once;
  Factory factory;
  std::shared_ptr<Runtime> runtime;
};

using Accessor = std::function<Runtime *()>;

class RuntimeRegistry {
public:
  static RuntimeRegistry *instance() {
    static RuntimeRegistry reg;
    return &reg;
  }

  void registerFactory(llvm::StringRef id, Factory factory) {
    std::lock_guard<std::mutex> lock{mutex};
    auto memoizer = std::make_shared<Memoizer>(std::move(factory));
    auto inserted =
        runtimes
            .emplace(id, Accessor{[memoizer = std::move(memoizer)]() {
                       return memoizer->get();
                     }})
            .second;
    if (!inserted) {
      throw DuplicateRuntimeError{id};
    }
  }

  void bindRuntime(llvm::StringRef id, std::shared_ptr<Runtime> runtime) {
    std::lock_guard<std::mutex> lock{mutex};
    auto inserted =
        runtimes
            .emplace(id,
                     [runtime = std::move(runtime)]() { return runtime.get(); })
            .second;
    if (!inserted) {
      throw DuplicateRuntimeError{id};
    }
  }

  Runtime *getRuntime(llvm::StringRef id) {
    std::lock_guard<std::mutex> lock{mutex};
    return runtimes.at(id.str())();
  }

  std::unordered_map<std::string, Runtime *> getRuntimeMap() {
    std::lock_guard<std::mutex> lock{mutex};
    std::unordered_map<std::string, Runtime *> result;
    for (auto &[id, runtimeFunc] : runtimes) {
      Runtime *runtime;
      try {
        runtime = runtimeFunc();
      } catch (...) {
        continue;
      }
      result[id] = runtime;
    }
    return result;
  }

private:
  std::mutex mutex;
  std::unordered_map<std::string, Accessor> runtimes;
};

} // namespace

namespace details {

void registerFactory(llvm::StringRef id, Factory factory) {
  RuntimeRegistry::instance()->registerFactory(id, std::move(factory));
}

} // namespace details

Runtime *getRuntime(llvm::StringRef id) {
  return RuntimeRegistry::instance()->getRuntime(id);
}

std::unordered_map<std::string, Runtime *> getRuntimeMap() {
  return RuntimeRegistry::instance()->getRuntimeMap();
}

void bindRuntime(llvm::StringRef id, std::shared_ptr<Runtime> runtime) {
  RuntimeRegistry::instance()->bindRuntime(id, std::move(runtime));
}

} // namespace pmlc::rt
