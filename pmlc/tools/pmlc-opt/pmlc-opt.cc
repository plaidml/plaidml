// Copyright 2020 Intel Corporation

#include "llvm/Support/Process.h"

#include "mlir/Support/MlirOptMain.h"

#include "pmlc/all_dialects.h"
#include "pmlc/all_passes.h"
#include "pmlc/util/logging.h"

int main(int argc, char **argv) {
  auto verboseEnv = llvm::sys::Process::GetEnv("PLAIDML_VERBOSE");
  if (verboseEnv) {
    auto level = std::atoi(verboseEnv->c_str());
    if (level) {
      el::Loggers::setVerboseLevel(level);
    }
    IVLOG(level, "PLAIDML_VERBOSE=" << level);
  }

  registerAllDialects();
  registerAllPasses();

  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  return failed(mlir::MlirOptMain(argc, argv, "PMLC modular optimizer driver\n",
                                  registry,
                                  /*preloadDialectsInContext=*/false));
}
