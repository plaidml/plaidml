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
//
// The implementation also supports specifying an expected runtime_error being
// thrown and validates with no failure the expected string is correctly
// thrown.
//===----------------------------------------------------------------------===//

#include <iostream>
#include <stdexcept>
#include <string>

#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Support/JitRunner.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include "pmlc/util/all_dialects.h"
#include "pmlc/util/all_passes.h"

#define EXPECTED_EXCEPTION_FLG_START_LEN 35
#define EXPECTED_EXCEPTION_FLAG_START "-expected-runtime-exception-string="

int main(int argc, char **argv) {
  registerAllDialects();
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::initializeLLVMPasses();

  // Get the expected-runtime-exception-string and remove the parameters
  // unknown to the tool that is being delegated to.
  std::string expectedString;
  int fixedUpCounter = 0;

  for (int i = 0; i < argc; i++) {
    if (fixedUpCounter != i) {
      argv[fixedUpCounter] = argv[i];
    }

    std::string arg = argv[i];
    if (arg.find(EXPECTED_EXCEPTION_FLAG_START) == 0) {
      expectedString = arg.substr(EXPECTED_EXCEPTION_FLG_START_LEN);
      continue;
    }

    fixedUpCounter++;
  }

  // Zero up the fixed parameters that have been moved forward.
  for (int i = fixedUpCounter; i < argc; i++) {
    argv[i] = nullptr;
  }

  try {
    return mlir::JitRunnerMain(fixedUpCounter, argv, nullptr);
  } catch (std::runtime_error e) {
    if (e.what() != expectedString) {
      throw e;
    }
  }

  return 0;
}
