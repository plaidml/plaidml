#pragma once

#include <google/protobuf/any.pb.h>

#include <exception>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "base/context/context.h"
#include "base/util/any_factory.h"
#include "base/util/type_url.h"

namespace vertexai {

// Instances of AnyFactoryMap provide coordination points between
// class-specific instance factories and classes that look up and use those factories.
template <typename Product>
class AnyFactoryMap final {
 public:
  // Returns the global instance for the specified product.
  static AnyFactoryMap<Product>* Instance();

  explicit AnyFactoryMap(const std::string& type_url_prefix_);

  // Adds an AnyFactory to the map.
  // Throws std::invalid_argument if there's already an AnyFactory for
  // the supplied factory's type URL.  This method is not synchronized.
  void Register(std::unique_ptr<AnyFactory<Product>> factory);

  // Instantiates a product instance from the supplied configuration.
  // Throws std::out_of_range if there is no factory able to process
  // the configuration, as well as exceptions thrown from the product-specific
  // construction.
  std::unique_ptr<Product> MakeInstance(const context::Context& ctx, const google::protobuf::Any& config) const;

  // Identical to MakeInstance, but returns an empty pointer if the config
  // is not supported by the factory.  Exceptions raised by the product-specific
  // construction may still be thrown.
  std::unique_ptr<Product> MakeInstanceIfSupported(const context::Context& ctx,
                                                   const google::protobuf::Any& config) const;

 private:
  std::string type_url_prefix_;
  std::unordered_map<std::string, std::unique_ptr<AnyFactory<Product>>> factories_;
};

template <typename Product>
AnyFactoryMap<Product>* AnyFactoryMap<Product>::Instance() {
  // N.B. function-local static objects are guaranteed to be initialized
  // exactly once, the first time the function is called.  This works well at
  // global initialization time, when the various global initializers might run
  // in any order.
  static AnyFactoryMap<Product> instance{kTypeVertexAIPrefix};
  return &instance;
}

template <typename Product>
AnyFactoryMap<Product>::AnyFactoryMap(const std::string& type_url_prefix) : type_url_prefix_{type_url_prefix} {}

template <typename Product>
void AnyFactoryMap<Product>::Register(std::unique_ptr<AnyFactory<Product>> factory) {
  std::string name = type_url_prefix_ + factory->full_name();
  bool emplaced;
  std::tie(std::ignore, emplaced) = factories_.emplace(name, std::move(factory));
  if (!emplaced) {
    throw std::range_error(std::string("duplicate factory for type: ") + name);
  }
}

template <typename Product>
std::unique_ptr<Product> AnyFactoryMap<Product>::MakeInstance(const context::Context& ctx,
                                                              const google::protobuf::Any& config) const {
  auto result = MakeInstanceIfSupported(ctx, config);
  if (!result) {
    throw std::out_of_range(std::string("unable to resolve type: ") + config.type_url());
  }
  return result;
}

template <typename Product>
std::unique_ptr<Product> AnyFactoryMap<Product>::MakeInstanceIfSupported(const context::Context& ctx,
                                                                         const google::protobuf::Any& config) const {
  auto it = factories_.find(config.type_url());
  if (it == factories_.end()) {
    return std::unique_ptr<Product>();
  }
  return it->second->MakeInstance(ctx, config);
}

}  // namespace vertexai
