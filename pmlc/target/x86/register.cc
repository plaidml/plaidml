#include "pmlc/target/x86/passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "pmlc/compiler/registry.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

static PassPipelineRegistration<>
    passPipelineReg("target-cpu", "Target pipeline for CPU", pipelineBuilder);

class Target final : public compiler::Target {
 public:
  void addPassesToPipeline(mlir::OpPassManager* mgr) {
    pipelineBuilder(*mgr);
  }
};

static compiler::TargetRegistration targetReg("llvm_cpu", std::make_shared<Target>());

} // namespace pmlc::target::x86
