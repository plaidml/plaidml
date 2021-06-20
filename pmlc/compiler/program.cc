// Copyright 2020 Intel Corporation

#include "pmlc/compiler/program.h"

#include <utility>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/FileUtilities.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::compiler {

namespace {

static bool isHiddenPass(Pass *pass, Operation *op) {
  if (pass->getName().startswith("mlir::detail::"))
    return true;
  if (auto funcOp = dyn_cast<FuncOp>(op)) {
    if (funcOp.isExternal())
      return true;
    std::string filter = pmlc::util::getEnvVar("PLAIDML_FUNC_FILTER");
    if (!filter.empty() && funcOp.getName() != filter)
      return true;
  }
  return false;
}

class IRCollector : public PassInstrumentation {
public:
  explicit IRCollector(std::vector<PassInfo> *into) : into(into) {}

private:
  void runAfterPass(Pass *pass, Operation *op) override {
    if (isHiddenPass(pass, op))
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

Program::Program(ModuleOp module)
    : context(std::make_unique<MLIRContext>()), module(module) {}

Program::Program(llvm::StringRef name)
    : context(std::make_unique<MLIRContext>()),
      module(ModuleOp::create(UnknownLoc::get(context.get()), name)) {}

std::unique_ptr<Program>
Program::fromSource(std::unique_ptr<MLIRContext> context, StringRef source) {
  return std::make_unique<Program>(std::move(context),
                                   llvm::MemoryBuffer::getMemBuffer(source));
}

Program::Program(std::unique_ptr<MLIRContext> context,
                 std::unique_ptr<llvm::MemoryBuffer> buffer)
    : context(std::move(context)) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  module = parseSourceFile(sourceMgr, this->context.get());
}

static StringRef getDiagKindStr(DiagnosticSeverity kind) {
  switch (kind) {
  case DiagnosticSeverity::Note:
    return "note";
  case DiagnosticSeverity::Warning:
    return "warning";
  case DiagnosticSeverity::Error:
    return "error";
  case DiagnosticSeverity::Remark:
    return "remark";
  }
  llvm_unreachable("Unknown DiagnosticSeverity");
}

void Program::compile(StringRef targetNameAndOptions, bool collectPasses,
                      StringRef dumpDir) {
  if (targetNameAndOptions.empty()) {
    return;
  }

  PassManager pm(module->getContext());
  ScopedDiagnosticHandler diagHandler(pm.getContext(), [&](Diagnostic &diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Error ||
        diag.getSeverity() == DiagnosticSeverity::Warning) {
      llvm::errs() << getDiagKindStr(diag.getSeverity()) << ": " << diag
                   << "\n";
      for (const auto &note : diag.getNotes()) {
        llvm::errs() << "  note: " << note << "\n";
      }
    }
    IVLOG(2, getDiagKindStr(diag.getSeverity()).str() << ": " << diag.str());
    for (const auto &note : diag.getNotes()) {
      IVLOG(2, "  note: " << note.str());
    }
    return success();
  });

  if (collectPasses || dumpDir.size()) {
    std::string ir;
    llvm::raw_string_ostream os(ir);
    module->print(os);
    passes.emplace_back(PassInfo{"tile", os.str()});
    pm.addInstrumentation(std::make_unique<IRCollector>(&passes));
    pm.getContext()->disableMultithreading();
  }

  if (VLOG_IS_ON(2)) {
    pm.enableStatistics();
    pm.enableTiming();
    auto shouldPrintBeforePass = [](auto *pass, auto *op) { return false; };
    auto shouldPrintAfterPass = [&](auto *pass, auto *op) {
      if (isHiddenPass(pass, op)) {
        return false;
      }
      return VLOG_IS_ON(3);
    };
    pm.getContext()->disableMultithreading();
    pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass,
                        /*printModuleScope=*/true,
                        /*printAfterOnlyOnChange=*/false,
                        /*printAfterOnlyOnFailure=*/false,
                        /*out=*/llvm::errs());
  }

  auto begOpts = targetNameAndOptions.find('{');
  auto targetName = targetNameAndOptions.substr(0, begOpts);
  auto targetOptions = targetNameAndOptions.substr(begOpts);

  // if target options are specified
  if (!targetOptions.empty()) {
    // trim off curly braces
    auto endOpts = targetOptions.find('}');
    targetOptions = targetOptions.substr(1, endOpts - 1);
  }

  target = resolveTarget(targetName);
  target->buildPipeline(pm, targetOptions);
  if (failed(pm.run(*module))) {
    throw std::runtime_error("Compilation failure");
  }

  if (dumpDir.size()) {
    std::string err;
    auto errCode = llvm::sys::fs::create_directories(dumpDir);
    if (errCode) {
      throw std::runtime_error("Could not create dumpDir: " +
                               errCode.message());
    }
    for (auto pass : llvm::enumerate(passes)) {
      const auto &info = pass.value();
      SmallString<128> path(dumpDir);
      llvm::sys::path::append(
          path, llvm::formatv("{0,0+2}_{1}.mlir", pass.index(), info.name));
      auto file = openOutputFile(path, &err);
      if (!err.empty()) {
        throw std::runtime_error("Failed to dump pass: " + err);
      }
      file->os() << info.ir;
      file->keep();
    }
  }
}

util::BufferPtr
Program::save(const std::unordered_map<std::string, std::string> &config) {
  if (!target) {
    throw std::runtime_error("Program must be compiled to be saved.");
  }
  return target->save(*this, config);
}

} // namespace pmlc::compiler
