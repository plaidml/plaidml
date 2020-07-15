//===- mlir-vulkan-runner.cpp - MLIR Vulkan Execution Driver --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that executes an MLIR file on the Vulkan by
// translating MLIR GPU module to SPIR-V and host part to LLVM IR before
// JIT-compiling and executing the latter.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include "pmlc/compiler/executable.h"
#include "pmlc/compiler/program.h"
#include "pmlc/conversion/gpu/lowering.h"
#include "pmlc/util/all_dialects.h"
#include "pmlc/util/all_passes.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]
using pmlc::compiler::Executable;
using pmlc::compiler::Program;

static LogicalResult runMLIRPasses(ModuleOp module) {
  PassManager passManager(module.getContext());
  applyPassManagerCLOptions(passManager);

  passManager.addPass(createGpuKernelOutliningPass());
  passManager.addPass(createLegalizeStdOpsForSPIRVLoweringPass());
  passManager.addPass(createConvertGPUToSPIRVPass());
  OpPassManager &modulePM = passManager.nest<spirv::ModuleOp>();
  modulePM.addPass(spirv::createLowerABIAttributesPass());
  modulePM.addPass(spirv::createUpdateVersionCapabilityExtensionPass());
  passManager.addPass(
      pmlc::conversion::gpu::createConvertGpuLaunchFuncToVulkanCallsPass());
  passManager.addPass(createLowerToLLVMPass(LowerToLLVMOptions{
      /*useBarePtrCallConv=*/false,
      /*emitCWrappers=*/true,
      /*indexBitwidth=*/kDeriveIndexBitwidthFromDataLayout,
      /*useAlignedAlloc=*/false,
  }));
  return passManager.run(module);
}

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

  runMLIRPasses(*program->module);

  Executable executable(program, ArrayRef<void *>{});
  executable.invoke();

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  auto level_str = pmlc::util::getEnvVar("PLAIDML_VERBOSE");
  if (level_str.size()) {
    auto level = std::atoi(level_str.c_str());
    if (level) {
      el::Loggers::setVerboseLevel(level);
    }
    IVLOG(level, "PLAIDML_VERBOSE=" << level);
  }

  llvm::llvm_shutdown_obj x;
  registerPassManagerCLOptions();

  registerAllDialects();
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::initializeLLVMPasses();

  return JitRunnerMain(argc, argv);
}
