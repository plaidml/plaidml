// Copyright 2020 Intel Corporation

#include "pmlc/compiler/program.h"

#include <utility>

#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::compiler {

namespace {

class IRCollector : public PassInstrumentation {
public:
  explicit IRCollector(std::vector<PassInfo> *into) : into(into) {}

private:
  bool isHiddenPass(Pass *pass) {
    return pass->getName().startswith("detail::");
  }

  void runAfterPass(Pass *pass, Operation *op) override {
    if (isHiddenPass(pass))
      return;

    std::string ir;
    llvm::raw_string_ostream os(ir);

    // Find the top-level module operation.
    auto *topLevelOp = op;
    while (auto *parentOp = topLevelOp->getParentOp()) {
      topLevelOp = parentOp;
    }

    // Check to see if the top-level operation is actually a module in the case
    // of invalid-ir.
    if (auto module = dyn_cast<ModuleOp>(topLevelOp)) {
      module.print(os);
    } else {
      topLevelOp->print(os);
    }

    os.flush();

    auto name = pass->getName().str();
    if (auto passInfo = pass->lookupPassInfo()) {
      auto passArg = passInfo->getPassArgument();
      if (!passArg.empty()) {
        name = passArg.str();
      }
    }
    into->emplace_back(PassInfo{name, ir});
  }

  std::vector<PassInfo> *into;
};

} // namespace

Program::Program(mlir::ModuleOp module) : module(module) {}

Program::Program(mlir::StringRef source) {
  auto inputBuffer = llvm::MemoryBuffer::getMemBuffer(source);
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(inputBuffer), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);
}

void Program::compile(StringRef target, bool collectPasses) {
  if (target.empty()) {
    return;
  }

  PassManager pm(module->getContext());

  if (collectPasses) {
    std::string ir;
    llvm::raw_string_ostream os(ir);
    module->print(os);
    os.flush();
    passes.emplace_back(PassInfo{"tile", ir});
    pm.addInstrumentation(std::make_unique<IRCollector>(&passes));
    pm.disableMultithreading();
  }

  if (VLOG_IS_ON(1)) {
    pm.enableStatistics();
    pm.enableTiming();
    auto shouldPrintBeforePass = [](auto pass, auto op) { return false; };
    auto shouldPrintAfterPass = [&](auto pass, auto op) {
      return VLOG_IS_ON(3);
    };
    pm.disableMultithreading();
    pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass, true,
                        false, llvm::errs());
  }

  auto pipelineBuilder = resolveTarget(target);
  pipelineBuilder(pm);

  if (failed(pm.run(*module))) {
    throw std::runtime_error("conversion to the LLVM IR dialect failed");
  }
}

} // namespace pmlc::compiler
