// Copyright 2018, Intel Corporation

#pragma once

#include <memory>

#include "base/util/any_factory.h"
#include "base/util/any_factory_map.h"
#include "tile/base/buffer.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct MLIRState;

struct CompilerState {
  explicit CompilerState(std::shared_ptr<stripe::Program> prog);
  ~CompilerState();

  std::unique_ptr<MLIRState> mlir;
  std::shared_ptr<stripe::Program> prog;
  ConstBufferManager* const_bufs;

  stripe::Block* entry() { return prog->entry.get(); }
};

// A CompilePass applies an arbitrary transformation to a Stripe
// program, rewriting it in place.
class CompilePass {
 public:
  virtual ~CompilePass() {}
  virtual bool is_stripe() const { return true; }
  virtual void Apply(CompilerState* root) const = 0;
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
