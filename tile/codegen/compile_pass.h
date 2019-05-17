// Copyright 2018, Intel Corporation

#pragma once

#include <memory>

#include "base/util/any_factory.h"
#include "base/util/any_factory_map.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

// A CompilePass applies an arbitrary transformation to a Stripe
// program, rewriting it in place.
class CompilePass {
 public:
  virtual ~CompilePass() {}
  virtual void Apply(stripe::Block* root) const = 0;
};

// CompilePassFactory implements the TypedAnyFactory abstraction by
// forwarding instance creation to a pass-specific constructor.
template <typename Impl, typename Config>
class CompilePassFactory : public TypedAnyFactory<CompilePass, Config> {
 public:
  std::unique_ptr<CompilePass> MakeTypedInstance(const context::Context& /* ctx */, const Config& config) final {
    return std::make_unique<Impl>(config);
  }

  static void Register() {
    AnyFactoryMap<CompilePass>::Instance()->Register(std::make_unique<CompilePassFactory<Impl, Config>>());
  }
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
