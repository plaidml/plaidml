// Copyright 2020, Intel Corporation

#include "pmlc/rt/runtime_registry.h"

#include <list>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "llvm/Support/FormatVariadic.h"

namespace pmlc::runtime {
namespace {

struct LoaderList {
  std::mutex mutex;
  std::list<Loader> loaders;

  static LoaderList &instance() {
    static LoaderList ll;
    return ll;
  }
};

class MemoizedFactory final {
public:
  explicit MemoizedFactory(Factory factory) : factory{std::move(factory)} {}
  Runtime *getRuntime() {
    std::call_once(init_once, [this]() { runtime = factory(); });
    return runtime.get();
  }

private:
  std::once_flag init_once;
  Factory factory;
  std::shared_ptr<Runtime> runtime;
};

const std::unordered_map<std::string, std::function<Runtime *()>>
buildRuntimeMap() {
  std::unordered_map<std::string, std::function<Runtime *()>> result;

  auto &ll = LoaderList::instance();
  std::lock_guard<std::mutex> lock{ll.mutex};

  for (auto &loader : ll.loaders) {
    for (auto &[id, factory] : loader()) {
      auto memo = std::make_shared<MemoizedFactory>(factory);
      auto [it, inserted] =
          result.emplace(id, [memo = std::move(memo)]() -> Runtime * {
            return memo->getRuntime();
          });
      if (!inserted) {
        throw std::runtime_error{
            llvm::formatv("Multiple runtime implementations found for {}", id)};
      }
    }
  }

  return result;
}

} // namespace

namespace details {

void registerLoader(Loader loader) {
  auto &ll = LoaderList::instance();
  std::lock_guard<std::mutex> lock{ll.mutex};
  ll.loaders.emplace_back(std::move(loader));
}

} // namespace details

const std::unordered_map<std::string, std::function<Runtime *()>> &
getRuntimeMap() {
  static std::unordered_map<std::string, std::function<Runtime *()>>
      runtimeMap = buildRuntimeMap();
  return runtimeMap;
}

} // namespace pmlc::runtime
