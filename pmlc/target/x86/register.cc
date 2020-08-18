#include "pmlc/target/x86/passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "pmlc/compiler/registry.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

static PassPipelineRegistration<>
    passPipelineReg("target-cpu", "Target pipeline for CPU", pipelineBuilder);
static compiler::TargetRegistration targetReg("llvm_cpu", pipelineBuilder);

} // namespace pmlc::target::x86
