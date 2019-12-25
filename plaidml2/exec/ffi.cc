// Copyright 2019 Intel Corporation.

#include "plaidml2/exec/ffi.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/Support/FormatVariadic.h"

#include "base/util/env.h"
#include "plaidml2/core/internal.h"
#include "tile/targets/targets.h"

#ifdef PLAIDML_AST
#include "tile/lang/gen_stripe.h"
#endif
#ifdef PLAIDML_MLIR
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/TargetSelect.h"

#include "pmlc/dialect/pxa/passes.h"
#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/transcode.h"
#include "pmlc/dialect/tile/lowering.h"
#include "pmlc/dialect/tile/program.h"
#endif

using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using plaidml::core::GetPlatform;
using plaidml::core::GlobalContext;
using vertexai::context::Context;
using vertexai::tile::Allocator;
using vertexai::tile::Buffer;
using vertexai::tile::BufferPtr;
using vertexai::tile::ConstBufferManager;
using vertexai::tile::Program;
using vertexai::tile::View;
using vertexai::tile::targets::GetConfigs;

#ifdef PLAIDML_AST
using vertexai::tile::lang::ast::ExprPtr;
using vertexai::tile::lang::ast::ParamExpr;
#endif
#ifdef PLAIDML_MLIR
using pmlc::dialect::pxa::createLowerToAffinePass;
using pmlc::dialect::pxa::createLowerToPXAPass;
using pmlc::dialect::tile::ProgramArgument;
using StripeDialect = pmlc::dialect::stripe::Dialect;
using pmlc::dialect::stripe::FromMLIR;
using pmlc::dialect::tile::LowerIntoStripe;
using namespace mlir;  // NOLINT[build/namespaces]
#endif

namespace {

class PlatformAllocator : public Allocator {
 public:
  explicit PlatformAllocator(const std::string& device_id) : device_id_(device_id) {}

  std::shared_ptr<Buffer> allocate(size_t size) {
    Context ctx;
    return GetPlatform()->MakeBuffer(ctx, device_id_, size);
  }

 private:
  std::string device_id_;
};

#ifdef PLAIDML_MLIR

template <typename T, int N>
struct StridedMemRefType {
  T* basePtr;
  T* data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

template <typename StreamType, typename T, int N>
void printMemRefMetaData(StreamType& os, StridedMemRefType<T, N>* memref) {  // NOLINT[runtime/references]
  static_assert(N > 0, "Expected N > 0");
  os << "Memref ptr: " << reinterpret_cast<void*>(memref);
  os << " base: " << reinterpret_cast<void*>(memref->data);
  os << " rank: " << N;
  os << " offset: " << memref->offset;
  os << " sizes: [";
  for (unsigned i = 0; i < N; ++i) {
    if (i) {
      os << ", ";
    }
    os << memref->sizes[i];
  }
  os << "] strides: [";
  for (unsigned i = 0; i < N; ++i) {
    if (i) {
      os << ", ";
    }
    os << memref->strides[i];
  }
  os << "]";
}

template <typename T, int N>
void printMemRef(StridedMemRefType<T, N>* memref) {
  static_assert(N > 0, "Expected N > 0");
  printMemRefMetaData(std::cout, memref);
  // std::cout << " data = " << std::endl;
  // MemRefDataPrinter<T, N>::print(std::cout, M.data, N, M.offset, M.sizes, M.strides);
  std::cout << std::endl;
}

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

struct Executable {
  std::string entry;
  std::unique_ptr<ExecutionEngine> engine;
  std::vector<MemRefDescriptor> descriptors;
  std::vector<void*> args;
  std::vector<void*> ptrs;

  Executable(StringRef entry, ModuleOp programModule, ArrayRef<ProgramArgument> programArgs);

  void invoke() {
    auto result = engine->invoke(entry, llvm::MutableArrayRef<void*>(args));
    if (result) {
      throw std::runtime_error("JIT invocation failed");
    }
  }
};

using MemRefTypes = std::vector<MemRefType>;

class ArgumentCollectorPass : public FunctionPass<ArgumentCollectorPass> {
 public:
  explicit ArgumentCollectorPass(MemRefTypes* into) : into(into) {}

  void runOnFunction() override {
    auto funcOp = getFunction();
    for (auto arg : funcOp.getArguments()) {
      into->emplace_back(arg->getType().cast<MemRefType>());
    }
  }

  static std::unique_ptr<Pass> create(MemRefTypes* into) { return std::make_unique<ArgumentCollectorPass>(into); }

 private:
  MemRefTypes* into;
};

class InjectTracingPass : public FunctionPass<InjectTracingPass> {
 public:
  void runOnFunction() override {
    auto funcOp = getFunction();
    auto moduleOp = funcOp.getParentOfType<ModuleOp>();

    OpBuilder builder(funcOp.getBody());
    for (auto arg : funcOp.getArguments()) {
      auto memRefType = arg->getType().cast<MemRefType>();
      SmallVector<int64_t, 2> shape(memRefType.getRank(), MemRefType::kDynamicSize);
      auto genericType = MemRefType::get(shape, memRefType.getElementType());
      auto printRef = getOrInsertPrint(moduleOp, genericType);
      auto castOp = builder.create<MemRefCastOp>(builder.getUnknownLoc(), genericType, arg);
      builder.create<CallOp>(builder.getUnknownLoc(), printRef, ArrayRef<Type>{}, castOp.getResult());
    }
  }

  static FlatSymbolRefAttr getOrInsertPrint(ModuleOp module, MemRefType memRefType) {
    auto* context = module.getContext();
    // TODO: select symbol name based on memRefType
    const char* symbol = "print_memref_2d_f32";
    if (module.lookupSymbol<FuncOp>(symbol)) {
      return SymbolRefAttr::get(symbol, context);
    }
    OpBuilder builder(context);
    builder.setInsertionPointToStart(module.getBody());
    auto funcType = FunctionType::get(memRefType, {}, context);
    builder.create<FuncOp>(module.getLoc(), symbol, funcType, ArrayRef<NamedAttribute>{});
    return SymbolRefAttr::get(symbol, context);
  }

  static std::unique_ptr<Pass> create() { return std::make_unique<InjectTracingPass>(); }
};

Executable::Executable(StringRef entry, ModuleOp programModule, ArrayRef<ProgramArgument> programArgs)
    : entry(entry), args(programArgs.size()), ptrs(programArgs.size()) {
  auto copy = cast<ModuleOp>(programModule.getOperation()->clone());
  OwningModuleRef module(copy);
  PassManager manager(module->getContext());
  auto shouldPrintBeforePass = [](auto, auto) { return false; };
  auto shouldPrintAfterPass = [](auto, auto) { return VLOG_IS_ON(3); };
  std::vector<MemRefType> memRefTypes;
  manager.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass, true, true, llvm::errs());
  manager.addNestedPass<FuncOp>(createCanonicalizerPass());
  manager.addNestedPass<FuncOp>(createCSEPass());
  manager.addPass(createLowerToPXAPass());
  manager.addNestedPass<FuncOp>(createCanonicalizerPass());
  manager.addNestedPass<FuncOp>(createCSEPass());
  manager.addPass(createLowerToAffinePass());
  manager.addNestedPass<FuncOp>(createCanonicalizerPass());
  manager.addNestedPass<FuncOp>(createCSEPass());
  manager.addPass(createLowerAffinePass());
  manager.addNestedPass<FuncOp>(createCanonicalizerPass());
  manager.addNestedPass<FuncOp>(createCSEPass());
  // manager.addPass(createLowerToCFGPass());
  // manager.addNestedPass<FuncOp>(createCanonicalizerPass());
  // manager.addNestedPass<FuncOp>(createCSEPass());
  manager.addPass(ArgumentCollectorPass::create(&memRefTypes));
  if (VLOG_IS_ON(6)) {
    manager.addPass(InjectTracingPass::create());
  }
  manager.addPass(createLowerToLLVMPass(true));
  if (failed(manager.run(*module))) {
    throw std::runtime_error("conversion to the LLVM IR dialect failed");
  }

  assert(memRefTypes.size() == programArgs.size() && "memRefTypes and programArgs size mismatch");

  auto optPipeline = makeOptimizingTransformer(
      /*optLevel=*/0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

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

  auto ctx = GlobalContext::getContext();
  for (unsigned i = 0; i < args.size(); i++) {
    auto view = programArgs[i].buffer->MapCurrent(*ctx).get();
    descriptors.emplace_back(view->data(), memRefTypes[i]);
    ptrs[i] = descriptors[i].ptr();
    args[i] = &ptrs[i];
  }
}

std::vector<ProgramArgument> BindProgramArguments(  //
    plaidml_program* program,                       //
    size_t ninputs,                                 //
    plaidml_binding** inputs,                       //
    size_t noutputs,                                //
    plaidml_binding** outputs) {
  std::unordered_map<Value*, BufferPtr> input_bindings;
  for (unsigned i = 0; i < ninputs; i++) {
    input_bindings[inputs[i]->expr->value] = inputs[i]->buffer->buffer;
  }
  std::unordered_map<Value*, BufferPtr> output_bindings;
  for (unsigned i = 0; i < noutputs; i++) {
    output_bindings[outputs[i]->expr->value] = outputs[i]->buffer->buffer;
  }
  std::vector<ProgramArgument> args(program->program->arguments.size());
  for (unsigned i = 0; i < args.size(); i++) {
    auto arg = program->program->arguments[i];
    if (arg.isInput) {
      auto it = input_bindings.find(arg.value);
      if (it != input_bindings.end()) {
        arg.buffer = it->second;
      }
      IVLOG(1, " Input[" << i << "]: " << arg.buffer);
      if (!arg.buffer) {
        throw std::runtime_error("Unbound input");
      }
    } else {
      auto it = output_bindings.find(arg.value);
      if (it != output_bindings.end()) {
        arg.buffer = it->second;
      }
      IVLOG(1, "Output[" << i << "]: " << arg.buffer);
      if (!arg.buffer) {
        throw std::runtime_error("Unbound output");
      }
    }
    args[i] = arg;
  }
  return args;
}

#endif  // PLAIDML_MLIR

}  // namespace

extern "C" {

#ifdef PLAIDML_MLIR

void print_memref_2d_f32(StridedMemRefType<float, 2>* M) { printMemRef(M); }

#endif  // PLAIDML_MLIR

struct plaidml_executable {
  using BufferMap = std::map<std::string, std::shared_ptr<Buffer>>;
  BufferMap input_bufs;
  BufferMap output_bufs;
  std::shared_ptr<Program> program;
#ifdef PLAIDML_MLIR
  std::unique_ptr<Executable> exec;
#endif  // PLAIDML_MLIR
};

void plaidml_exec_init(  //
    plaidml_error* err) {
  static std::once_flag is_initialized;
  ffi_wrap_void(err, [&] {
    std::call_once(is_initialized, []() {
      IVLOG(1, "plaidml_exec_init");
      GetPlatform();
#ifdef PLAIDML_MLIR
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      initializeLLVMPasses();
#endif  // PLAIDML_MLIR
    });
  });
}

plaidml_strings* plaidml_devices_get(  //
    plaidml_error* err) {
  return ffi_wrap<plaidml_strings*>(err, nullptr, [&] {
    auto devices = GetPlatform()->ListDevices();
    auto strs = new plaidml_string*[devices.size()];
    for (size_t i = 0; i < devices.size(); i++) {
      strs[i] = new plaidml_string{devices[i]};
    }
    return new plaidml_strings{devices.size(), strs};
  });
}

plaidml_strings* plaidml_targets_get(  //
    plaidml_error* err) {
  return ffi_wrap<plaidml_strings*>(err, nullptr, [&] {
    auto configs = GetConfigs().configs();
    auto strs = new plaidml_string*[configs.size()];
    size_t i = 0;
    for (const auto& [key, value] : configs) {
      strs[i++] = new plaidml_string{key};
    }
    return new plaidml_strings{configs.size(), strs};
  });
}

plaidml_executable* plaidml_compile(  //
    plaidml_error* err,               //
    plaidml_program* program,         //
    const char* device,               //
    const char* target,               //
    size_t ninputs,                   //
    plaidml_binding** inputs,         //
    size_t noutputs,                  //
    plaidml_binding** outputs) {
  return ffi_wrap<plaidml_executable*>(err, nullptr, [&] {
    IVLOG(1, "Compiling with device: " << device << ", target: " << target);
    auto configs = GetConfigs().configs();
    if (!configs.count(target)) {
      throw std::runtime_error(llvm::formatv("Unknown target specified: {0}", target).str());
    }
#ifdef PLAIDML_AST
    ConstBufferManager const_bufs;
    const_bufs.allocator = std::make_shared<PlatformAllocator>(device);
    auto exec = std::make_unique<plaidml_executable>();
    auto stripe = vertexai::tile::lang::GenerateStripe(program->eval.runinfo);
    Context ctx;
    exec->program = GetPlatform()->MakeProgram(ctx, device, target, stripe, &const_bufs);
    std::unordered_map<ExprPtr, BufferPtr> input_bindings;
    for (size_t i = 0; i < ninputs; i++) {
      auto param_expr = std::dynamic_pointer_cast<ParamExpr>(inputs[i]->expr->expr);
      if (!param_expr) {
        throw std::runtime_error("Buffers can only be bound to ParamExprs");
      }
      param_expr->buffer = inputs[i]->buffer->buffer;
      input_bindings[inputs[i]->expr->expr] = inputs[i]->buffer->buffer;
    }
    std::unordered_map<ExprPtr, BufferPtr> output_bindings;
    for (size_t i = 0; i < noutputs; i++) {
      output_bindings[outputs[i]->expr->expr] = outputs[i]->buffer->buffer;
    }
    for (const auto& arg : program->eval.args) {
      if (arg.is_input) {
        auto it = input_bindings.find(arg.expr);
        auto param_expr = std::dynamic_pointer_cast<ParamExpr>(arg.expr);
        auto buffer = (it == input_bindings.end()) ? param_expr->buffer : it->second;
        exec->input_bufs[arg.name] = buffer;
      } else {
        auto it = output_bindings.find(arg.expr);
        if (it == output_bindings.end()) {
          throw std::runtime_error("Invalid program, unbound output");
        }
        exec->output_bufs[arg.name] = it->second;
      }
    }
    for (const auto& kvp : program->eval.updates) {
      exec->output_bufs[kvp.first] = kvp.second->buffer;
    }
    return exec.release();
#endif
#ifdef PLAIDML_MLIR
    auto args = BindProgramArguments(program, ninputs, inputs, noutputs, outputs);
    if (vertexai::env::Get("PLAIDML_EE") == "1") {
      auto exec = std::make_unique<plaidml_executable>();
      exec->exec = std::make_unique<Executable>(program->program->entry, *program->program->module, args);
      return exec.release();
    }
    ConstBufferManager const_bufs;
    const_bufs.allocator = std::make_shared<PlatformAllocator>(device);
    auto exec = std::make_unique<plaidml_executable>();

    // 1. lower tile dialect -> stripe dialect
    auto module = LowerIntoStripe(*program->program->module);
    // 2. convert MLIR -> stripe
    auto stripe = FromMLIR(*module);
    Context ctx;
    exec->program = GetPlatform()->MakeProgram(ctx, device, target, stripe, &const_bufs);
    IVLOG(1, "After make program");

    auto attrName = StripeDialect::getDialectAttrName("name");
    auto stripeFuncOp = cast<FuncOp>(module->getBody()->front());
    for (unsigned i = 0; i < args.size(); i++) {
      const auto& arg = args[i];
      auto attr = stripeFuncOp.getArgAttrOfType<StringAttr>(i, attrName);
      if (!attr) {
        throw std::runtime_error("Missing expected argument attribute");
      }
      auto name = attr.getValue().str();
      if (arg.isInput) {
        exec->input_bufs[name] = arg.buffer;
      } else {
        exec->output_bufs[name] = arg.buffer;
      }
    }

    return exec.release();
#endif
  });
}

void plaidml_executable_free(  //
    plaidml_error* err,        //
    plaidml_executable* exec) {
  ffi_wrap_void(err, [&] {  //
    delete exec;
  });
}

void plaidml_executable_run(  //
    plaidml_error* err,       //
    plaidml_executable* exec) {
  ffi_wrap_void(err, [&] {
#ifdef PLAIDML_MLIR
    if (exec->exec) {
      exec->exec->invoke();
    } else {
#endif  // PLAIDML_MLIR
      auto ctx = GlobalContext::getContext();
      exec->program->Run(*ctx, exec->input_bufs, exec->output_bufs).get();
#ifdef PLAIDML_MLIR
    }
#endif  // PLAIDML_MLIR
  });
}

}  // extern "C"
