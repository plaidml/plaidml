// Copyright 2018, Intel Corporation

#include <functional>

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "tile/codegen/compile_pass.h"
#include "tile/codegen/driver.h"

#include "pmlc/dialect/stripe/padding_pass.h"

namespace vertexai {
namespace tile {
namespace codegen {

template <class Pass, class Config>
class MlirCompilePass : public CompilePass {
 public:
  bool is_stripe() const override { return false; }
  explicit MlirCompilePass(const Config& cfg) : config(cfg) {}
  void Apply(CompilerState* root) const override {
    mlir::PassManager pm;
    pm.addPass(mlir::createCSEPass());
    pm.addPass(new Pass(config));
    if (failed(pm.run(root->module))) {
      throw std::runtime_error("Failed to run pass\n");
    }
  }

 private:
  Config config;
};

#define REGISTER(pass, config)                                                                    \
  [[gnu::unused]] char reg_##pass = []() -> char {                                                \
    CompilePassFactory<MlirCompilePass<pmlc::dialect::stripe::pass, config>, config>::Register(); \
    return 0;                                                                                     \
  }();

namespace {

REGISTER(PaddingPass, proto::MLIR_PadPass);

}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
