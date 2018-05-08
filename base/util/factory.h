#pragma once

#include <map>
#include <memory>

#include "base/context/context.h"

namespace vertexai {

template <typename Product>
class FactoryRegistrar final {
 public:
  typedef std::function<std::unique_ptr<Product>(const context::Context& ctx)> Factory;

  static FactoryRegistrar<Product>* Instance() {
    static FactoryRegistrar<Product> instance;
    return &instance;
  }

  void Register(Factory factory, int priority = 0) { factories_.insert(std::make_pair(priority, factory)); }

  std::multimap<int, Factory> Factories() { return factories_; }

 private:
  std::multimap<int, Factory> factories_;
};

}  // namespace vertexai
