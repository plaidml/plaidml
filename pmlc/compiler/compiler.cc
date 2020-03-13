// Copyright 2020 Intel Corporation

#include "pmlc/compiler/compiler.h"

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

#include "pmlc/compiler/registry.h"
#include "pmlc/rt/vulkan/vulkan_runtime.h"
#include "pmlc/util/all_dialects.h"
#include "pmlc/util/all_passes.h"
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

void Program::initialize() {
  registerAllDialects();
  registerAllPasses();
}

void Executable::initialize() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  initializeLLVMPasses();
}

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
  targetValue = target;
}

Executable::Executable(const std::shared_ptr<Program> &program,
                       ArrayRef<void *> bufptrs)
    : program(program), ptrs(bufptrs.size()) {
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
      /*optLevel=*/0, /*sizeLevel=*/0,
      /*targetMachine=*/tmOrError->get());

  if (VLOG_IS_ON(6)) {
    auto llvmModule = translateModuleToLLVMIR(*program->module);
    if (!llvmModule) {
      throw std::runtime_error("could not convert to LLVM IR");
    }
    llvmModule->print(llvm::errs(), nullptr);
  }

  auto maybeEngine = ExecutionEngine::create(*program->module, optPipeline);
  llvm::handleAllErrors(
      maybeEngine.takeError(), [](const llvm::ErrorInfoBase &err) {
        throw std::runtime_error("Failed to create ExecutionEngine: " +
                                 err.message());
      });
  engine = std::move(*maybeEngine);

  descriptors.reserve(bufptrs.size());
  for (unsigned i = 0; i < bufptrs.size(); i++) {
    IVLOG(1, "buffer " << bufptrs[i]);
    descriptors.emplace_back(bufptrs[i], program->arguments[i].shape);
    ptrs[i] = descriptors[i].ptr();
  }
}

Executable::~Executable() = default;

void Executable::invoke() {
  IVLOG(1, "Executable target:" << program->targetValue);
  if (program->targetValue == "intel_gen") {
    IVLOG(1, "Do setResourceData here");
    for (size_t i = 0; i < ptrs.size(); i++) {
      IVLOG(1, "buffer:" << (reinterpret_cast<void **>(ptrs[i]))[0])
      setResourceData1(0, i, (reinterpret_cast<int64_t **>(ptrs[i]))[0],
                       (reinterpret_cast<int64_t **>(ptrs[i]))[0], 0,
                       program->arguments[i].shape.getNumElements(),
                       program->arguments[i].shape.getRank());
    }
  }

  auto arrayRef = MutableArrayRef<void *>(ptrs);
  auto result = engine->invoke(program->entry, arrayRef);
  if (result) {
    throw std::runtime_error("JIT invocation failed");
  }
}

} // namespace pmlc::compiler
