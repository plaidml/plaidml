// Copyright 2020 Intel Corporation

#include "pmlc/rt/jit_executable.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
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
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/rt/device_id.h"
#include "pmlc/rt/instrument.h"
#include "pmlc/rt/internal.h"
#include "pmlc/rt/runtime.h"
#include "pmlc/rt/symbol_registry.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT[build/namespaces]

using pmlc::compiler::Program;

namespace pmlc::rt {

namespace {

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

static std::string makePackedFunctionName(StringRef name) {
  return "_mlir_" + name.str();
}

static std::string makeCWrapperFunctionName(StringRef name) {
  return "_mlir_ciface_" + name.str();
}

// Define an interface function that wraps all the arguments of the original
// function and all its results into an i8** pointer to provide a unified
// invocation interface.
static std::string packFunctionArguments(llvm::Module *module,
                                         StringRef entry) {
  auto funcName = makeCWrapperFunctionName(entry);
  auto newName = makePackedFunctionName(entry);
  util::wrapFunctionAndPackArguments(module, funcName, newName);
  return newName;
}

constexpr const char *kInitName = "init";
constexpr const char *kFiniName = "fini";

class MemRefDescriptor {
private:
  struct Base {
    void *basePtr;
    void *data;
    int64_t offset;
    int64_t sizesAndStrides[];
  };

public:
  MemRefDescriptor(void *data, Type type)
      : rankedType(type.cast<RankedTensorType>()),
        memory(computeSize(rankedType)) {
    auto base = reinterpret_cast<Base *>(memory.data());
    base->basePtr = data;
    base->data = data;
    auto rank = rankedType.getRank();
    auto shape = rankedType.getShape();
    auto memRefType = MemRefType::get(shape, rankedType.getElementType());
    SmallVector<int64_t, 8> strides;
    (void)getStridesAndOffset(memRefType, strides, base->offset);
    for (unsigned i = 0; i < rank; i++) {
      base->sizesAndStrides[i] = shape[i];
      base->sizesAndStrides[i + rank] = strides[i];
    }
  }

  void *ptr() { return memory.data(); }

  void set(void *data) {
    auto base = reinterpret_cast<Base *>(memory.data());
    base->basePtr = data;
    base->data = data;
  }

private:
  static unsigned computeSize(RankedTensorType type) {
    return sizeof(void *) +                   // allocatedPtr
           sizeof(void *) +                   // alignedPtr
           sizeof(int64_t) +                  // offset
           sizeof(int64_t) * type.getRank() + // sizes
           sizeof(int64_t) * type.getRank();  // strides
  }

  RankedTensorType rankedType;
  std::vector<char> memory;
};

using Function = uint8_t *(*)(void **);

struct EngineImpl {
  virtual ~EngineImpl() = default;
  virtual void compile(std::unique_ptr<llvm::Module> module,
                       std::unique_ptr<llvm::LLVMContext> ctx) = 0;
  virtual Function getFunction(StringRef symbol) = 0;
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

  void compile(std::unique_ptr<llvm::Module> module,
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
  }

  Function getFunction(StringRef symbol) final {
    uint64_t addr = engine->getFunctionAddress(symbol.str());
    if (!addr) {
      throw std::runtime_error(
          llvm::formatv("Entry point not found: {0}", symbol.str()));
    }
    return reinterpret_cast<Function>(addr);
  }

  std::unique_ptr<llvm::ExecutionEngine> engine;
};

struct OrcJITEngineImpl : EngineImpl {
  void compile(std::unique_ptr<llvm::Module> module,
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

    auto tmBuilderOrError = JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError) {
      throw std::runtime_error(
          "Failed to create a JITTargetMachineBuilder for the host");
    }

    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError) {
      throw std::runtime_error("Failed to create a TargetMachine for the host");
    }

    auto transformer = makeOptimizingTransformer(
        /*optLevel=*/3,
        /*sizeLevel=*/0,
        /*targetMachine=*/tmOrError->get());

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
    cantFail(tsm.withModuleDo(
        [&](llvm::Module &module) { return transformer(&module); }));

    if (VLOG_IS_ON(6)) {
      tsm.withModuleDo(
          [&](llvm::Module &module) { module.print(llvm::errs(), nullptr); });
    }

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
  }

  Function getFunction(StringRef symbol) {
    // JIT lookup may return an Error referring to strings stored internally by
    // the JIT. If the Error outlives the ExecutionEngine, it would want have a
    // dangling reference, which is currently caught by an assertion inside JIT
    // thanks to hand-rolled reference counting. Rewrap the error message into a
    // string before returning. Alternatively, ORC JIT should consider copying
    // the string into the error message.
    auto expectedSymbol = jit->lookup(symbol);
    if (!expectedSymbol) {
      std::string errorMessage;
      llvm::raw_string_ostream os(errorMessage);
      llvm::handleAllErrors(expectedSymbol.takeError(),
                            [&os](llvm::ErrorInfoBase &ei) { ei.log(os); });
      throw std::runtime_error(os.str());
    }

    auto addr = expectedSymbol->getAddress();
    return reinterpret_cast<Function>(addr);
  }

  std::unique_ptr<llvm::orc::LLJIT> jit;
};

enum class EngineKind {
  MCJIT,
  OrcJIT,
};

class JitExecutable final : public Executable {
public:
  JitExecutable(const std::shared_ptr<Program> &program,
                std::shared_ptr<Device> device, ArrayRef<void *> preParams)
      : program(program), device(std::move(device)),
        preParams(preParams.begin(), preParams.end()) {
    static std::once_flag is_initialized;
    std::call_once(is_initialized, []() {
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      initializeLLVMPasses();
      llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
    });
    registerLLVMDialectTranslation(*program->context);
    registerOpenMPDialectTranslation(*program->context);

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

    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto llvmModule = translateModuleToLLVMIR(*program->module, *ctx);
    if (!llvmModule) {
      throw std::runtime_error("could not convert to LLVM IR");
    }

    setupTargetTriple(llvmModule.get());
    auto entryPoint = packFunctionArguments(llvmModule.get(), program->entry);
    std::string initPacked;
    std::string finiPacked;
    if (llvmModule->getFunction(kInitName) &&
        llvmModule->getFunction(kFiniName)) {
      initPacked = packFunctionArguments(llvmModule.get(), kInitName);
      finiPacked = packFunctionArguments(llvmModule.get(), kFiniName);
    }

    if (VLOG_IS_ON(6)) {
      llvmModule->print(llvm::errs(), nullptr);
    }

    impl->compile(std::move(llvmModule), std::move(ctx));
    jitMain = impl->getFunction(entryPoint);
    if (initPacked != "" && finiPacked != "") {
      jitInit = impl->getFunction(initPacked);
      jitFini = impl->getFunction(finiPacked);
    }

    if (!jitMain) {
      throw std::runtime_error("jitEntry function is null");
    }

    if (jitInit) {
      IVLOG(3, "Doing jit init");
      std::vector<void *> initPtrs;
      std::vector<MemRefDescriptor> initDescriptors;
      std::copy(preParams.begin(), preParams.end(),
                std::back_inserter(initPtrs));
      for (const compiler::ConstantArgument &arg : program->constants) {
        initDescriptors.emplace_back(arg.buffer->data(), arg.type);
        initPtrs.push_back(initDescriptors.back().ptr());
      }
      rt::initInstrument();
      initPack = jitInit(initPtrs.data());
      IVLOG(3, "Jit init complete");

      ptrs.push_back(initPack);
      for (Type type : program->inputs) {
        descriptors.emplace_back(nullptr, type);
        ptrs.push_back(descriptors.back().ptr());
      }
      for (Type type : program->outputs) {
        descriptors.emplace_back(nullptr, type);
        ptrs.push_back(descriptors.back().ptr());
      }
    } else {
      std::copy(preParams.begin(), preParams.end(), std::back_inserter(ptrs));
      for (Type type : program->inputs) {
        descriptors.emplace_back(nullptr, type);
        ptrs.push_back(descriptors.back().ptr());
      }
      for (const compiler::ConstantArgument &arg : program->constants) {
        descriptors.emplace_back(arg.buffer->data(), arg.type);
        ptrs.push_back(descriptors.back().ptr());
      }
      for (Type type : program->outputs) {
        descriptors.emplace_back(nullptr, type);
        ptrs.push_back(descriptors.back().ptr());
      }
    }
  }

  ~JitExecutable() {
    if (jitFini) {
      IVLOG(3, "Doing jit fini");
      rt::initInstrument();
      SmallVector<void *, 1> finiPtrs{initPack};
      jitFini(finiPtrs.data());
      IVLOG(3, "Jit fini complete");
      free(initPack);
    }
  }

  double invoke() final {
    std::vector<util::BufferPtr> inputBuffers, outputBuffers;

    for (Type type : program->inputs) {
      auto shape = util::TensorShape::fromType(type);
      auto buffer = std::make_shared<util::SimpleBuffer>(shape);
      inputBuffers.emplace_back(buffer);
    }

    for (Type type : program->outputs) {
      auto shape = util::TensorShape::fromType(type);
      auto buffer = std::make_shared<util::SimpleBuffer>(shape);
      outputBuffers.emplace_back(buffer);
    }

    return invoke(inputBuffers, outputBuffers);
  }

  double invoke(ArrayRef<util::BufferPtr> inputBuffers,
                ArrayRef<util::BufferPtr> outputBuffers) final {
    rt::initInstrument();
    bindArguments(inputBuffers, outputBuffers);
    jitMain(ptrs.data());
    return device->execTimeInMS;
  }

  void bindArguments(ArrayRef<util::BufferPtr> inputBuffers,
                     ArrayRef<util::BufferPtr> outputBuffers) {
    if (inputBuffers.size() != program->inputs.size()) {
      throw std::runtime_error("Program input arguments and buffers mismatch");
    }

    if (outputBuffers.size() != program->outputs.size()) {
      throw std::runtime_error(
          "Program outputs arguments and buffers mismatch");
    }

    unsigned i = 0;
    if (jitInit) {
      for (util::BufferPtr buffer : inputBuffers) {
        descriptors[i++].set(buffer->data());
      }
      for (util::BufferPtr buffer : outputBuffers) {
        descriptors[i++].set(buffer->data());
      }
    } else {
      for (util::BufferPtr buffer : inputBuffers) {
        descriptors[i++].set(buffer->data());
      }
      i += program->constants.size();
      for (util::BufferPtr buffer : outputBuffers) {
        descriptors[i++].set(buffer->data());
      }
    }
  }

private:
  std::shared_ptr<Program> program;
  std::shared_ptr<Device> device;
  SmallVector<void *> preParams;
  std::unique_ptr<EngineImpl> impl;
  SmallVector<MemRefDescriptor> descriptors;
  SmallVector<void *> ptrs;
  Function jitInit = nullptr;
  Function jitMain = nullptr;
  Function jitFini = nullptr;
  uint8_t *initPack = nullptr;
};

} // namespace

std::unique_ptr<Executable>
makeJitExecutable(const std::shared_ptr<Program> &program,
                  std::shared_ptr<Device> device, ArrayRef<void *> preParams) {
  return std::make_unique<JitExecutable>(program, std::move(device), preParams);
}

} // namespace pmlc::rt
