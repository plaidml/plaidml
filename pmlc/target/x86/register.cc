#include "pmlc/target/x86/passes.h"

#include "llvm/Support/FormatVariadic.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "pmlc/compiler/registry.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

static constexpr const char *kTargetName = "llvm_cpu";
static constexpr const char *kPassPipelineTargetName = "target-llvm_cpu";

static PassPipelineRegistration<> passPipelineReg(kPassPipelineTargetName,
                                                  "Target pipeline for CPU",
                                                  pipelineBuilder);

class Target : public compiler::Target {
public:
  void buildPipeline(mlir::OpPassManager &pm) { pipelineBuilder(pm); }

  util::BufferPtr save(compiler::Program &program) {
    throw std::runtime_error(
        llvm::formatv("Target '{0}' does not have 'save' support.", kTargetName)
            .str());
  }
};

static compiler::TargetRegistration targetReg(kTargetName, []() {
  return std::make_shared<Target>();
});

} // namespace pmlc::target::x86
