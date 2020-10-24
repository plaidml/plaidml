// Copyright 2020 Intel Corporation

#include "pmlc/rt/jit_executable.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/rt/device_id.h"
#include "pmlc/rt/internal.h"
#include "pmlc/rt/runtime.h"
#include "pmlc/rt/symbol_registry.h"
#include "pmlc/util/env.h"
#include "pmlc/util/ids.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

using pmlc::compiler::Program;

namespace pmlc::rt {

// Setup LLVM target triple from the current machine.
static void setupTargetTriple(llvm::Module *llvmModule) {
  // Setup the machine properties from the current architecture.
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target) {
    throw std::runtime_error("NO target: " + errorMessage);
  }

  std::string cpu(llvm::sys::getHostCPUName());
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> hostFeatures;

  if (llvm::sys::getHostCPUFeatures(hostFeatures))
    for (auto &f : hostFeatures)
      features.AddFeature(f.first(), f.second);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), {}, {}));
  llvmModule->setDataLayout(machine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);
}

namespace {

class MemRef {
public:
  struct Descriptor {
    void *basePtr;
    void *data;
    int64_t offset;
    int64_t sizesAndStrides[];
  };

  explicit MemRef(RankedTensorType type) : memory(computeSize(type)) {
    auto descriptor = getDescriptor();
    descriptor->basePtr = nullptr;
    descriptor->data = nullptr;
    auto rank = type.getRank();
    auto shape = type.getShape();
    auto memRefType = MemRefType::get(shape, type.getElementType());
    SmallVector<int64_t, 8> strides;
    getStridesAndOffset(memRefType, strides, descriptor->offset);
    for (unsigned i = 0; i < rank; i++) {
      descriptor->sizesAndStrides[i] = shape[i];
      descriptor->sizesAndStrides[i + rank] = strides[i];
    }
  }

  Descriptor *getDescriptor() {
    return reinterpret_cast<Descriptor *>(memory.data());
  }

  void setDataPtr(void *data) {
    auto descriptor = getDescriptor();
    descriptor->basePtr = data;
    descriptor->data = data;
  }

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

struct Network;
using SetupFunc = Network *(*)(Device *device);
using ExecuteFunc = void (*)(Network *, MemRef::Descriptor **descriptors);
using TeardownFunc = void (*)(Network *);

struct ABI {
  SetupFunc setupFunc = nullptr;
  ExecuteFunc executeFunc = nullptr;
  TeardownFunc teardownFunc = nullptr;

  bool operator!() { return !setupFunc || !executeFunc || !teardownFunc; }
};

struct EngineImpl {
  virtual ~EngineImpl() = default;
  virtual ABI compile(std::unique_ptr<llvm::Module> module,
                      std::unique_ptr<llvm::LLVMContext> ctx) = 0;
};

static void *tryResolveSymbol(StringRef symbol) {
  if (auto ptr = resolveSymbol(symbol))
    return ptr;
  if (symbol[0] == '_') {
    if (auto ptr = resolveSymbol(symbol.drop_front()))
      return ptr;
  }

  auto ptr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(symbol.str());
  if (ptr)
    return ptr;

  if (symbol[0] == '_') {
    if (auto ptr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(
            symbol.drop_front().str())) {
      return ptr;
    }
  }

  return nullptr;
}

struct MCJITEngineImpl : EngineImpl {
  struct Runtime : public llvm::LegacyJITSymbolResolver {
    llvm::JITSymbol findSymbol(const std::string &symbol) override {
      auto ptr = tryResolveSymbol(symbol);
      if (!ptr) {
        throw std::runtime_error(
            llvm::formatv("Could not find symbol: {0}", symbol));
      }
      auto addr = llvm::pointerToJITTargetAddress(ptr);
      return llvm::JITEvaluatedSymbol(addr, llvm::JITSymbolFlags::None);
    }

    llvm::JITSymbol findSymbolInLogicalDylib(const std::string &) override {
      return llvm::JITSymbol(nullptr);
    }
  };

  ABI compile(std::unique_ptr<llvm::Module> module,
              std::unique_ptr<llvm::LLVMContext> ctx) final {
    std::string error;
    std::unique_ptr<llvm::LegacyJITSymbolResolver> resolver(new Runtime);
    engine = std::unique_ptr<llvm::ExecutionEngine>(
        llvm::EngineBuilder(std::move(module))
            .setErrorStr(&error)
            .setOptLevel(llvm::CodeGenOpt::Aggressive)
            .setEngineKind(llvm::EngineKind::JIT)
            .setVerifyModules(true)
            .setSymbolResolver(std::move(resolver))
            .create());
    if (!engine) {
      throw std::runtime_error("Failed to create ExecutionEngine: " + error);
    }

    engine->finalizeObject();

    ABI abi;
    abi.setupFunc = getFunc<SetupFunc>(util::kPlaidmlInit);
    abi.executeFunc = getFunc<ExecuteFunc>(util::kPlaidmlExec);
    abi.teardownFunc = getFunc<TeardownFunc>(util::kPlaidmlFini);
    return abi;
  }

  template <typename Func>
  Func getFunc(const std::string &name) {
    uint64_t addr = engine->getFunctionAddress(name);
    if (!addr) {
      throw std::runtime_error(
          llvm::formatv("Entry point not found: {0}", name));
    }
    return reinterpret_cast<Func>(addr);
  }

  std::unique_ptr<llvm::ExecutionEngine> engine;
};

struct OrcJITEngineImpl : EngineImpl {
  ABI compile(std::unique_ptr<llvm::Module> module,
              std::unique_ptr<llvm::LLVMContext> ctx) final {
    using llvm::Expected;
    using llvm::orc::DynamicLibrarySearchGenerator;
    using llvm::orc::IRCompileLayer;
    using llvm::orc::JITTargetMachineBuilder;
    using llvm::orc::MangleAndInterner;
    using llvm::orc::SymbolMap;
    using llvm::orc::ThreadSafeModule;
    using llvm::orc::TMOwningSimpleCompiler;

    auto dataLayout = module->getDataLayout();

    // Callback to inspect the cache and recompile on demand. This follows
    // Lang's LLJITWithObjectCache example.
    auto compileFunctionCreator = [&](JITTargetMachineBuilder JTMB)
        -> Expected<std::unique_ptr<IRCompileLayer::IRCompiler>> {
      JTMB.setCodeGenOptLevel(llvm::CodeGenOpt::Level::Aggressive);
      auto TM = JTMB.createTargetMachine();
      if (!TM)
        return TM.takeError();
      return std::make_unique<TMOwningSimpleCompiler>(std::move(*TM), nullptr);
    };

    jit = llvm::cantFail(llvm::orc::LLJITBuilder()
                             .setCompileFunctionCreator(compileFunctionCreator)
                             .create());

    // Add a ThreadSafeModule to the engine and return.
    ThreadSafeModule tsm(std::move(module), std::move(ctx));
    llvm::cantFail(jit->addIRModule(std::move(tsm)));

    SymbolMap symbols;
    auto &session = jit->getExecutionSession();

    auto addSymbol = [&](StringRef name, void *ptr) {
      auto addr = llvm::pointerToJITTargetAddress(ptr);
      auto symbol = llvm::JITEvaluatedSymbol(addr, llvm::JITSymbolFlags::None);
      symbols.insert(std::make_pair(session.intern(name), symbol));
    };

    for (const auto &kvp : SymbolRegistry::instance()->symbols) {
      addSymbol(kvp.first(), kvp.second);
#ifdef __APPLE__
      addSymbol(llvm::formatv("_{0}", kvp.first()).str(), kvp.second);
#endif
    }

    auto &mainJitDylib = jit->getMainJITDylib();
    cantFail(mainJitDylib.define(absoluteSymbols(symbols)));
    mainJitDylib.addGenerator(
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
            dataLayout.getGlobalPrefix())));

    ABI abi;
    abi.setupFunc = getFunc<SetupFunc>(util::kPlaidmlInit);
    abi.executeFunc = getFunc<ExecuteFunc>(util::kPlaidmlExec);
    abi.teardownFunc = getFunc<TeardownFunc>(util::kPlaidmlFini);
    return abi;
  }

  template <typename Func>
  Func getFunc(mlir::StringRef name) {
    // JIT lookup may return an Error referring to strings stored internally by
    // the JIT. If the Error outlives the ExecutionEngine, it would want have a
    // dangling reference, which is currently caught by an assertion inside JIT
    // thanks to hand-rolled reference counting. Rewrap the error message into a
    // string before returning. Alternatively, ORC JIT should consider copying
    // the string into the error message.
    auto expectedSymbol = jit->lookup(name);
    if (!expectedSymbol) {
      std::string errorMessage;
      llvm::raw_string_ostream os(errorMessage);
      llvm::handleAllErrors(expectedSymbol.takeError(),
                            [&os](llvm::ErrorInfoBase &ei) { ei.log(os); });
      throw std::runtime_error(os.str());
    }

    auto addr = expectedSymbol->getAddress();
    return reinterpret_cast<Func>(addr);
  }

  std::unique_ptr<llvm::orc::LLJIT> jit;
};

enum class EngineKind {
  MCJIT,
  OrcJIT,
};

struct StopWatch {
  using fp_milliseconds =
      std::chrono::duration<double, std::chrono::milliseconds::period>;

  void start() { startTime = std::chrono::steady_clock::now(); }

  void stop() { stopTime = std::chrono::steady_clock::now(); }

  double delta_ms() {
    return std::chrono::duration_cast<fp_milliseconds>(stopTime - startTime)
        .count();
  }

  std::chrono::steady_clock::time_point startTime;
  std::chrono::steady_clock::time_point stopTime;
};

class JitExecutable final : public Executable {
public:
  JitExecutable(const std::shared_ptr<Program> &program,
                std::shared_ptr<Device> device)
      : program(program), device(std::move(device)) {
    static std::once_flag is_initialized;
    std::call_once(is_initialized, []() {
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      initializeLLVMPasses();
      llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
    });

    IVLOG(3, "In JitExecutable::JitExecutable");

    EngineKind kind = EngineKind::OrcJIT;
    auto jit = pmlc::util::getEnvVar("LLVM_JIT");
    if (jit == "ORC") {
      kind = EngineKind::OrcJIT;
    } else if (jit == "MCJIT") {
      kind = EngineKind::MCJIT;
    }

    switch (kind) {
    case EngineKind::MCJIT:
      impl = std::make_unique<MCJITEngineImpl>();
      break;
    case EngineKind::OrcJIT:
      impl = std::make_unique<OrcJITEngineImpl>();
      break;
    default:
      throw std::runtime_error("Invalid EngineKind");
    }

    IVLOG(3, "Translating to LLVMIR");

    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto llvmModule = translateModuleToLLVMIR(*program->module, *ctx);
    if (!llvmModule) {
      throw std::runtime_error("could not convert to LLVM IR");
    }
    IVLOG(3, "Setting target triple");

    setupTargetTriple(llvmModule.get());

    if (VLOG_IS_ON(3)) {
      llvmModule->print(llvm::errs(), nullptr);
    }

    IVLOG(3, "Compiling");
    abi = impl->compile(std::move(llvmModule), std::move(ctx));
    if (!abi) {
      throw std::runtime_error("Entrypoint functions not found");
    }

    IVLOG(3, "Building memref descriptors");
    buildMemRefDescriptors();
    IVLOG(3, "Calling setup");
    network = abi.setupFunc(device.get());
    if (!network) {
      throw std::runtime_error("Unable to initialize the network");
    }
    IVLOG(3, "Compiled");
  }

  ~JitExecutable() {
    if (network) {
      abi.teardownFunc(network);
    }
  }

  void invoke(mlir::ArrayRef<util::BufferPtr> inputBuffers,
              mlir::ArrayRef<util::BufferPtr> outputBuffers) final {
    if (inputBuffers.size() != program->inputs.size()) {
      throw std::runtime_error("Program input arguments and buffers mismatch");
    }
    if (outputBuffers.size() != program->outputs.size()) {
      throw std::runtime_error(
          "Program outputs arguments and buffers mismatch");
    }
    StopWatch stopWatch;
    if (VLOG_IS_ON(1)) {
      stopWatch.start();
    }
    auto memrefIt = memrefs.begin();
    for (auto &bp : inputBuffers) {
      (memrefIt++)->setDataPtr(bp->data());
    }
    std::advance(memrefIt, program->constants.size());
    for (auto &bp : outputBuffers) {
      (memrefIt++)->setDataPtr(bp->data());
    }
    IVLOG(3, "Running executable");
    abi.executeFunc(network, descriptors.data());
    IVLOG(3, "Executable complete");
    if (VLOG_IS_ON(1)) {
      stopWatch.stop();
      IVLOG(1, "Execution time: " << stopWatch.delta_ms() << "ms");
    }
  }

  void buildMemRefDescriptors() {
    for (auto type : program->inputs) {
      memrefs.emplace_back(type.cast<RankedTensorType>());
      descriptors.push_back(memrefs.back().getDescriptor());
    }
    for (const compiler::ConstantArgument &arg : program->constants) {
      memrefs.emplace_back(arg.type.cast<RankedTensorType>());
      descriptors.push_back(memrefs.back().getDescriptor());
    }
    for (auto type : program->outputs) {
      memrefs.emplace_back(type.cast<RankedTensorType>());
      descriptors.push_back(memrefs.back().getDescriptor());
    }
  }

private:
  std::shared_ptr<Program> program;
  std::shared_ptr<Device> device;
  std::unique_ptr<EngineImpl> impl;
  std::vector<MemRef> memrefs;
  std::vector<MemRef::Descriptor *> descriptors;
  ABI abi;
  Network *network = nullptr;
};

} // namespace

std::unique_ptr<Executable>
makeJitExecutable(const std::shared_ptr<Program> &program,
                  std::shared_ptr<Device> device) {
  return std::make_unique<JitExecutable>(program, std::move(device));
}

} // namespace pmlc::rt
