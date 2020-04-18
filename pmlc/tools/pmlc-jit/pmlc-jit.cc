//===- pmlc-jit.cpp - PMLC CPU Execution Driver----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry point to a command line utility that executes an MLIR file on the
// CPU by translating MLIR to LLVM IR before JIT-compiling and executing the
// latter.
//===----------------------------------------------------------------------===//

#include <memory>
#include <stdexcept>

#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "pmlc/compiler/executable.h"
#include "pmlc/compiler/program.h"
#include "pmlc/util/all_dialects.h"
#include "pmlc/util/all_passes.h"

using llvm::Error;
using pmlc::compiler::Executable;
using pmlc::compiler::Program;

namespace {
/// This options struct prevents the need for global static initializers, and
/// is only initialized if the JITRunner is invoked.
struct Options {
  llvm::cl::opt<std::string> inputFilename{llvm::cl::Positional,
                                           llvm::cl::desc("<input file>"),
                                           llvm::cl::init("-")};
  llvm::cl::opt<std::string> mainFuncName{
      "e", llvm::cl::desc("The function to be called"),
      llvm::cl::value_desc("<function name>"), llvm::cl::init("main")};
};
} // namespace

int JitRunnerMain(int argc, char **argv) {
  // Create the options struct containing the command line options for the
  // runner. This must come before the command line options are parsed.
  Options options;
  llvm::cl::ParseCommandLineOptions(argc, argv, "pmlc execution driver\n");

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(options.inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return EXIT_FAILURE;
  }

  auto program = std::make_shared<Program>(std::move(file));
  program->entry = options.mainFuncName.getValue();
  Executable executable(program, ArrayRef<void *>{});
  executable.invoke();

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  registerAllDialects();
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::initializeLLVMPasses();

  std::set_terminate([]() {
    auto eptr = std::current_exception();
    if (eptr) {
      try {
        std::rethrow_exception(eptr);
      } catch (const std::exception &ex) {
        llvm::outs() << "ERROR: " << ex.what() << "\n";
      } catch (...) {
        llvm::outs() << "ERROR: Unknown exception\n";
      }
    } else {
      llvm::outs() << "Abnormal termination\n";
    }
    llvm::outs().flush();
    std::exit(EXIT_SUCCESS);
  });

  try {
    return JitRunnerMain(argc, argv);
  } catch (const std::exception &e) {
    llvm::outs() << "ERROR: " << e.what() << "\n";
    llvm::outs().flush();
  } catch (...) {
    llvm::outs() << "ERROR: Unknown exception\n";
    llvm::outs().flush();
  }

  return EXIT_SUCCESS;
}
