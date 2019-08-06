// Copyright 2018, Intel Corporation

#pragma once

#include <memory>

#include "base/util/any_factory.h"
#include "base/util/any_factory_map.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "tile/base/buffer.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

class CompilerState {
 public:
  explicit CompilerState(std::shared_ptr<stripe::Program> prog_) : prog(prog_), const_bufs(nullptr) {
    module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
  }

  // Always valid
  mlir::MLIRContext ctx;
  // Holds a single function or no function depending on if state is in MLIR
  mlir::ModuleOp module;

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
