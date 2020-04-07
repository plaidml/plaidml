// Copyright 2020 Intel Corporation

#include "pmlc/compiler/compiler.h"

#include <fstream>
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

    void **p_base = reinterpret_cast<void **>(memory.data());
    int64_t *p_params = reinterpret_cast<int64_t *>(p_base + 2);

    // TODO: set offset according to data
    p_params[0] = 0;

    // set size to memory
    auto rank = type.getRank();
    for (auto i = 0; i < rank; i++) {
      p_params[i + 1] = type.getDimSize(i);
    }

    // set stride to memory
    int64_t stride = 1;
    for (auto i = rank - 1; i >= 0; i--) {
      p_params[rank + 1 + i] = stride;
      stride *= type.getDimSize(i);
    }
  }

  void *ptr() { return memory.data(); }
  unsigned byteSize() { return memory.size(); }

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
  this->target = target;
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

  auto maybeEngine = ExecutionEngine::create(*program->module, optPipeline,
                                             /*jitCodeGenOptLevel=*/llvm::None,
                                             sharedLibPaths);
  llvm::handleAllErrors(
      maybeEngine.takeError(), [](const llvm::ErrorInfoBase &err) {
        throw std::runtime_error("Failed to create ExecutionEngine: " +
                                 err.message());
      });
  engine = std::move(*maybeEngine);

  descriptors.reserve(bufptrs.size());
  for (unsigned i = 0; i < bufptrs.size(); i++) {
    descriptors.emplace_back(bufptrs[i], program->arguments[i].shape);
    if (program->target == "llvm_cpu") {
      ptrs[i] = descriptors[i].ptr();
    } else if (program->target == "intel_gen") {
      // TODO: set createLowerToLLVMPass(false,true,false) so that llvm host
      // function takes memref as parameters
      void **p_base = reinterpret_cast<void **>(descriptors[i].ptr());
      int num_elements = descriptors[i].byteSize() / sizeof(void *);
      ptrs.resize(num_elements * bufptrs.size());
      for (int j = 0; j < num_elements; j++) {
        ptrs[num_elements * i + j] = p_base + j;
      }
    } else {
      throw std::runtime_error("unknown target");
    }
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
