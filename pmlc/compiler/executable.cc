// Copyright 2020 Intel Corporation

#include "pmlc/compiler/executable.h"

#include <fstream>
#include <string>
#include <utility>

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::compiler {

class MemRefDescriptor {
private:
  struct Base {
    void *basePtr;
    void *data;
    int64_t offset;
  };

public:
  MemRefDescriptor(void *data, RankedTensorType type)
      : memory(computeSize(type)) {
    auto base = reinterpret_cast<Base *>(memory.data());
    base->basePtr = data;
    base->data = data;
  }

  void *ptr() { return memory.data(); }

private:
  static unsigned computeSize(RankedTensorType type) {
    return sizeof(void *) +                   // allocatedPtr
           sizeof(void *) +                   // alignedPtr
           sizeof(int64_t) +                  // offset
           sizeof(int64_t) * type.getRank() + // sizes
           sizeof(int64_t) * type.getRank();  // strides
  }

  std::vector<char> memory;
};

Executable::Executable(const std::shared_ptr<Program> &program,
                       ArrayRef<void *> bufptrs)
    : program(program), ptrs(bufptrs.size()) {

  static std::once_flag is_initialized;
  std::call_once(is_initialized, []() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    initializeLLVMPasses();
  });

  if (program->arguments.size() != bufptrs.size()) {
    throw std::runtime_error("Program arguments and bufptrs size mismatch");
  }

  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    throw std::runtime_error(
        "Failed to create a JITTargetMachineBuilder for the host");
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    throw std::runtime_error("Failed to create a TargetMachine for the host");
  }

  auto optPipeline = makeOptimizingTransformer(
      /*optLevel=*/0,
      /*sizeLevel=*/0,
      /*targetMachine=*/tmOrError->get());

  if (VLOG_IS_ON(6)) {
    auto llvmModule = translateModuleToLLVMIR(*program->module);
    if (!llvmModule) {
      throw std::runtime_error("could not convert to LLVM IR");
    }
    llvmModule->print(llvm::errs(), nullptr);
  }

  std::vector<StringRef> sharedLibPaths;
  // HACK: this is required because the ORCv2 JIT doesn't want to resolve
  // symbols from the current process, but only on Linux.
#ifdef __linux__
  std::string modulePathHolder;
  {
    std::ifstream ifs("/proc/self/maps");
    if (!ifs.is_open()) {
      throw std::runtime_error("Could not load /proc/self/maps");
    }
    for (std::string line; std::getline(ifs, line);) {
      auto pos = line.find('/');
      if (pos == std::string::npos)
        continue;
      auto modulePath = line.substr(pos);
      pos = modulePath.find("libplaidml.so");
      if (pos != std::string::npos) {
        IVLOG(1, "module: " << modulePath);
        modulePathHolder = modulePath;
        sharedLibPaths.push_back(modulePathHolder);
        break;
      }
    }
  }
#endif

  auto maybeEngine =
      ExecutionEngine::create(*program->module, optPipeline,
                              /*jitCodeGenOptLevel=*/llvm::None, sharedLibPaths,
                              /*enableObjectCache=*/true,
                              /*enableGDBNotificationListener=*/false);
  llvm::handleAllErrors(
      maybeEngine.takeError(), [](const llvm::ErrorInfoBase &err) {
        throw std::runtime_error("Failed to create ExecutionEngine: " +
                                 err.message());
      });
  engine = std::move(*maybeEngine);

  descriptors.reserve(bufptrs.size());
  for (unsigned i = 0; i < bufptrs.size(); i++) {
    descriptors.emplace_back(bufptrs[i], program->arguments[i].shape);
    ptrs[i] = descriptors[i].ptr();
  }
}

Executable::~Executable() = default;

void Executable::invoke() {
  auto arrayRef = MutableArrayRef<void *>(ptrs);
  auto result = engine->invoke(program->entry, arrayRef);
  if (result) {
    throw std::runtime_error("JIT invocation failed");
  }
}

} // namespace pmlc::compiler
