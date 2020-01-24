// Copyright 2019, Intel Corporation

#include "pmlc/compiler/compiler.h"

#include <unordered_map>
#include <utility>

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/tile_to_pxa/tile_to_pxa.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/util/logging.h"

using namespace mlir;  // NOLINT[build/namespaces]
using pmlc::conversion::tile_to_pxa::createLowerTileToPXAPass;

namespace pmlc::compiler {

namespace {

using MemRefTypes = std::vector<MemRefType>;

class ArgumentCollectorPass : public FunctionPass<ArgumentCollectorPass> {
 public:
  explicit ArgumentCollectorPass(MemRefTypes* into) : into(into) {}

  void runOnFunction() override {
    auto funcOp = getFunction();
    for (auto arg : funcOp.getArguments()) {
      into->emplace_back(arg.getType().cast<MemRefType>());
    }
  }

  static std::unique_ptr<Pass> create(MemRefTypes* into) { return std::make_unique<ArgumentCollectorPass>(into); }

 private:
  MemRefTypes* into;
};

}  // namespace

class MemRefDescriptor {
 private:
  struct Base {
    void* basePtr;
    void* data;
    int64_t offset;
  };

 public:
  MemRefDescriptor(void* data, MemRefType type) : memory(computeSize(type)) {
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto maybeStrides = getStridesAndOffset(type, strides, offset);
    if (failed(maybeStrides)) {
      throw std::runtime_error("unexpected non-strided memref");
    }
    auto base = reinterpret_cast<Base*>(memory.data());
    base->basePtr = data;
    base->data = data;
    base->offset = offset;
    auto var = reinterpret_cast<int64_t*>(memory.data() + sizeof(Base));
    auto rank = type.getRank();
    auto sizes = type.getShape();
    for (unsigned i = 0; i < rank; i++) {
      var[i] = sizes[i];
      var[i + rank] = strides[i];
    }
  }

  void* ptr() { return memory.data(); }

 private:
  static unsigned computeSize(MemRefType type) {
    return sizeof(void*) +                     // allocatedPtr
           sizeof(void*) +                     // alignedPtr
           sizeof(int64_t) +                   // offset
           sizeof(int64_t) * type.getRank() +  // sizes
           sizeof(int64_t) * type.getRank();   // strides
  }

  std::vector<char> memory;
};

void Executable::initialize() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  initializeLLVMPasses();
}

Executable::Executable(StringRef entry, StringRef target, ModuleOp programModule, ArrayRef<void*> bufptrs,
                       FloatType floatx, IntegerType intx)
    : entry(entry), args(bufptrs.size()), ptrs(bufptrs.size()), floatx(floatx), intx(intx) {
  auto copy = cast<ModuleOp>(programModule.getOperation()->clone());
  OwningModuleRef module(copy);
  PassManager manager(module->getContext());

  auto shouldPrintBeforePass = [](auto, auto) { return false; };
  auto shouldPrintAfterPass = [](auto, auto) { return VLOG_IS_ON(3); };
  manager.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass, true, false, llvm::errs());

  manager.addPass(dialect::tile::createComputeBoundsPass());
  manager.addNestedPass<FuncOp>(createCanonicalizerPass());
  manager.addNestedPass<FuncOp>(createCSEPass());

  manager.addPass(createLowerTileToPXAPass());
  manager.addNestedPass<FuncOp>(createCanonicalizerPass());
  manager.addNestedPass<FuncOp>(createCSEPass());

  std::vector<MemRefType> memRefTypes;
  manager.addPass(ArgumentCollectorPass::create(&memRefTypes));

  auto pipelineBuilder = resolveTarget(target);
  pipelineBuilder(&manager);

  if (failed(manager.run(*module))) {
    throw std::runtime_error("conversion to the LLVM IR dialect failed");
  }

  assert(memRefTypes.size() == bufptrs.size() && "memRefTypes and bufptrs size mismatch");

  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    throw std::runtime_error("Failed to create a JITTargetMachineBuilder for the host");
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    throw std::runtime_error("Failed to create a TargetMachine for the host");
  }

  auto optPipeline = makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/tmOrError->get());

  if (VLOG_IS_ON(6)) {
    auto llvmModule = translateModuleToLLVMIR(*module);
    if (!llvmModule) {
      throw std::runtime_error("could not convert to LLVM IR");
    }
    llvmModule->print(llvm::errs(), nullptr);
  }

  auto maybeEngine = ExecutionEngine::create(*module, optPipeline);
  llvm::handleAllErrors(maybeEngine.takeError(), [](const llvm::ErrorInfoBase& b) {
    b.log(llvm::errs());
    throw std::runtime_error("Failed to create ExecutionEngine");
  });
  engine = std::move(*maybeEngine);

  descriptors.reserve(args.size());
  for (unsigned i = 0; i < args.size(); i++) {
    descriptors.emplace_back(bufptrs[i], memRefTypes[i]);
    ptrs[i] = descriptors[i].ptr();
    args[i] = &ptrs[i];
  }
}

Executable::~Executable() = default;

void Executable::invoke() {
  auto result = engine->invoke(entry, llvm::MutableArrayRef<void*>(args));
  if (result) {
    throw std::runtime_error("JIT invocation failed");
  }
}

}  // namespace pmlc::compiler
