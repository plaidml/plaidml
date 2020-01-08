// Copyright 2019 Intel Corporation.

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Analysis/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"

#include "base/util/env.h"
#include "base/util/logging.h"

using namespace llvm;  // NOLINT(build/namespaces)
using namespace mlir;  // NOLINT(build/namespaces)

static cl::opt<std::string> inputFilename(  //
    cl::Positional,                         //
    cl::desc("<input file>"),               //
    cl::init("-"));

static cl::opt<std::string> outputFilename(  //
    "o",                                     //
    cl::desc("Output filename"),             //
    cl::value_desc("filename"),              //
    cl::init("-"));

static cl::opt<bool> splitInputFile(                                                    //
    "split-input-file",                                                                 //
    cl::desc("Split the input file into pieces and process each chunk independently"),  //
    cl::init(false));

static cl::opt<bool> verifyDiagnostics(                                                           //
    "verify-diagnostics",                                                                         //
    cl::desc("Check that emitted diagnostics match expected-* lines on the corresponding line"),  //
    cl::init(false));

static cl::opt<bool> verifyPasses(                                //
    "verify-each",                                                //
    cl::desc("Run the verifier after each transformation pass"),  //
    cl::init(true));

int main(int argc, char** argv) {
  auto level_str = vertexai::env::Get("PLAIDML_VERBOSE");
  if (level_str.size()) {
    auto level = std::atoi(level_str.c_str());
    if (level) {
      el::Loggers::setVerboseLevel(level);
    }
  }

  PrettyStackTraceProgram x(argc, argv);
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
    errs() << errorMessage << "\n";
    return 1;
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    errs() << errorMessage << "\n";
    exit(1);
  }

  return failed(MlirOptMain(  //
      output->os(),           //
      std::move(file),        //
      passPipeline,           //
      splitInputFile,         //
      verifyDiagnostics,      //
      verifyPasses));
}
