#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "base/context/context.h"

namespace vertexai {

enum class FactoryPriority {
  HIGH = 100,
  DEFAULT = 0,
  LOW = -100,
};

template <typename Product>
class FactoryRegistrar final {
 public:
  typedef std::function<std::unique_ptr<Product>(const context::Context& ctx)> Factory;

  struct Entry {
    std::string name;
    Factory factory;
  };

  typedef std::multimap<FactoryPriority, Entry, std::greater<FactoryPriority>> FactoryMap;

  static FactoryRegistrar<Product>* Instance() {
    static FactoryRegistrar<Product> instance;
    return &instance;
  }

  void Register(const std::string& name,  //
                Factory factory,          //
                FactoryPriority priority = FactoryPriority::DEFAULT) {
    Entry entry{name, factory};
    factories_.insert(std::make_pair(priority, entry));
  }

  FactoryMap Factories() { return factories_; }

 private:
  FactoryMap factories_;
};

}  // namespace vertexai
