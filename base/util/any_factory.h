#pragma once

#include <google/protobuf/any.pb.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "base/context/context.h"

namespace vertexai {

// An AnyFactory is the type-generic interface implemented by TypedAnyFactory.
template <typename Product>
class AnyFactory {
 public:
  virtual ~AnyFactory() {}
  virtual const std::string& full_name() const = 0;
  virtual std::unique_ptr<Product> MakeInstance(const context::Context& ctx, const ::google::protobuf::Any& config) = 0;
};

// TypedAnyFactory is a type-specific class capable of unpacking an Any proto and invoking a type-specific creation
// method from a derived class.
template <typename Product, typename Config>
class TypedAnyFactory : public AnyFactory<Product> {
 public:
  // Returns the name of the configuration proto this factory uses to manufacture instances.
  const std::string& full_name() const final;

  // Manufactures an instance of the factory's product, according to the supplied configuration.
  std::unique_ptr<Product> MakeInstance(const context::Context& ctx, const ::google::protobuf::Any& config) final;

 protected:
  virtual std::unique_ptr<Product> MakeTypedInstance(const context::Context& ctx, const Config& config) = 0;
};

template <typename Product, typename Config>
const std::string& TypedAnyFactory<Product, Config>::full_name() const {
  return Config::descriptor()->full_name();
}

template <typename Product, typename Config>
std::unique_ptr<Product> TypedAnyFactory<Product, Config>::MakeInstance(const context::Context& ctx,
                                                                        const ::google::protobuf::Any& config) {
  Config typed_config;
  if (!config.UnpackTo(&typed_config)) {
    throw std::runtime_error(std::string("failed to unpack configuration"));
  }
  return MakeTypedInstance(ctx, typed_config);
}

}  // namespace vertexai
