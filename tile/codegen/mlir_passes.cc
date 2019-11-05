// Copyright 2019, Intel Corporation

#include "tile/codegen/mlir_passes.h"

#include <functional>
#include <memory>

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/dialect/eltwise/util.h"
#include "pmlc/dialect/stripe/nop_pass.h"
#include "pmlc/dialect/stripe/padding_pass.h"
#include "pmlc/dialect/stripe/transcode.h"
#include "tile/codegen/compile_pass.h"

// N.B. We need to confine all definitions to MLIR here.
// The reason is that when we mix Windows system headers with MLIR, we get
// compilation faliures. This is because MLIR currently uses the MSVC-specific
// reserved keyword 'interface' in some declarations.

namespace vertexai {
namespace tile {
namespace codegen {

struct MLIRState {
  // Always valid
  mlir::MLIRContext ctx;
  // Holds a single function or no function depending on if state is in MLIR
  mlir::OwningModuleRef module;

  MLIRState() : module(mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx))) {}
};

CompilerState::CompilerState(std::shared_ptr<stripe::Program> prog)
    : mlir(std::make_unique<MLIRState>()), prog(prog), const_bufs(nullptr) {}

CompilerState::~CompilerState() = default;

void ConvertFromMLIR(CompilerState* state) {
  IVLOG(1, "Converting from Stripe MLIR");
  *state->prog = *pmlc::dialect::stripe::FromMLIR(*state->mlir->module);
  IVLOG(3, "New\n" << *state->prog->entry);
}

void ConvertIntoMLIR(CompilerState* state) {
  IVLOG(1, "Converting to Stripe MLIR");
  IVLOG(3, "Original\n" << *state->prog->entry);
  state->mlir->module = pmlc::dialect::stripe::IntoMLIR(&state->mlir->ctx, *state->prog);
  auto module = *state->mlir->module;
  IVLOG(3, "New\n" << mlir::debugString(module));
}

template <typename Pass, typename Config>
std::unique_ptr<mlir::Pass> CreatePass(Config config) {
  return std::make_unique<Pass>(config);
}

template <class Pass, class Config>
class MlirCompilePass : public CompilePass {
 public:
  bool is_stripe() const override { return false; }
  explicit MlirCompilePass(const Config& cfg) : config(cfg) {}
  void Apply(CompilerState* root) const override {
    mlir::PassManager pm(&root->mlir->ctx, true);
    pm.addPass(mlir::createCSEPass());
    pm.addPass(CreatePass<Pass>(config));
    if (failed(pm.run(*root->mlir->module))) {
      throw std::runtime_error("Failed to run pass\n");
    }
  }

 private:
  Config config;
};

template <typename Pass, typename Config>
inline void RegisterPass() {
  CompilePassFactory<MlirCompilePass<Pass, Config>, Config>::Register();
}

[[gnu::unused]] char register_passes = []() -> char {
  RegisterPass<pmlc::dialect::stripe::NopPass, proto::MLIR_NopPass>();
  RegisterPass<pmlc::dialect::stripe::PaddingPass, proto::MLIR_PadPass>();
  return 0;
}();

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
