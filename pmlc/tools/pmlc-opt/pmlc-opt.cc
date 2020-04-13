// Copyright 2020 Intel Corporation

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"

#include "pmlc/util/all_dialects.h"
#include "pmlc/util/all_passes.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

using namespace llvm; // NOLINT(build/namespaces)
using namespace mlir; // NOLINT(build/namespaces)

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool> splitInputFile(
    "split-input-file",
    cl::desc("Split the input file into pieces and process each chunk "
             "independently"),
    cl::init(false));

static cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    cl::desc("Check that emitted diagnostics match expected-* lines on the "
             "corresponding line"),
    cl::init(false));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true));

static cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    cl::desc("Allow operation with no registered dialects"), cl::init(false));

int main(int argc, char **argv) {
  auto level_str = pmlc::util::getEnvVar("PLAIDML_VERBOSE");
  if (level_str.size()) {
    auto level = std::atoi(level_str.c_str());
    if (level) {
      el::Loggers::setVerboseLevel(level);
    }
    IVLOG(level, "PLAIDML_VERBOSE=" << level);
  }

  registerAllDialects();
  registerAllPasses();
  // registerTestPasses();
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  registerPassManagerCLOptions();
  PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "pmlc modular optimizer driver\n");

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  auto ret = failed(MlirOptMain(output->os(), std::move(file), passPipeline,
                                splitInputFile, verifyDiagnostics, verifyPasses,
                                allowUnregisteredDialects));
  llvm::outs() << "\n";
  if (ret) {
    return 1;
  }
  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return 0;
}
