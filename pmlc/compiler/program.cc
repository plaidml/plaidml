// Copyright 2020 Intel Corporation

#include "pmlc/compiler/program.h"

#include <utility>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/FileUtilities.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::compiler {

namespace {

static bool isHiddenPass(Pass *pass, Operation *op) {
  if (pass->getName().startswith("mlir::detail::")) {
    return true;
  }
  if (auto funcOp = dyn_cast<FuncOp>(op)) {
    if (funcOp.isExternal()) {
      return true;
    }
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

Program::Program(llvm::StringRef name)
    : module(ModuleOp::create(UnknownLoc::get(&context), name)) {}

Program::Program(mlir::ModuleOp module) : module(module) {}

std::unique_ptr<Program> Program::fromSource(mlir::StringRef source) {
  return std::make_unique<Program>(llvm::MemoryBuffer::getMemBuffer(source));
}

Program::Program(std::unique_ptr<llvm::MemoryBuffer> buffer) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);
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

void Program::compile(StringRef targetName, bool collectPasses,
                      StringRef dumpDir) {
  if (targetName.empty()) {
    return;
  }

  PassManager pm(module->getContext());
  ScopedDiagnosticHandler diagHandler(pm.getContext(), [&](Diagnostic &diag) {
    IVLOG(2, getDiagKindStr(diag.getSeverity()).str() << ": " << diag.str());
    for (auto &note : diag.getNotes()) {
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
                        /*out=*/llvm::errs());
  }

  target = resolveTarget(targetName);
  target->buildPipeline(pm);
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
      auto file = mlir::openOutputFile(path, &err);
      if (!err.empty()) {
        throw std::runtime_error("Failed to dump pass: " + err);
      }
      file->os() << info.ir;
      file->keep();
    }
  }
}

util::BufferPtr Program::save() {
  if (!target) {
    throw std::runtime_error("Program must be compiled to be saved.");
  }
  return target->save(*this);
}

} // namespace pmlc::compiler
