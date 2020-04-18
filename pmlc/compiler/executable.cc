// Copyright 2020 Intel Corporation

#include "pmlc/compiler/executable.h"

#include <fstream>
#include <string>
#include <utility>

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

static std::string makePackedFunctionName(StringRef name) {
  return "_mlir_" + name.str();
}

// For each function in the LLVM module, define an interface function that wraps
// all the arguments of the original function and all its results into an i8**
// pointer to provide a unified invocation interface.
static void packFunctionArguments(llvm::Module *module) {
  auto &ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  DenseSet<llvm::Function *> interfaceFunctions;
  for (auto &func : module->getFunctionList()) {
    if (func.isDeclaration()) {
      continue;
    }
    if (interfaceFunctions.count(&func)) {
      continue;
    }

    // Given a function `foo(<...>)`, define the interface function
    // `mlir_foo(i8**)`.
    auto newType = llvm::FunctionType::get(
        builder.getVoidTy(), builder.getInt8PtrTy()->getPointerTo(),
        /*isVarArg=*/false);
    auto newName = makePackedFunctionName(func.getName());
    auto funcCst = module->getOrInsertFunction(newName, newType);
    llvm::Function *interfaceFunc = cast<llvm::Function>(funcCst.getCallee());
    interfaceFunctions.insert(interfaceFunc);

    // Extract the arguments from the type-erased argument list and cast them to
    // the proper types.
    auto bb = llvm::BasicBlock::Create(ctx);
    bb->insertInto(interfaceFunc);
    builder.SetInsertPoint(bb);
    llvm::Value *argList = interfaceFunc->arg_begin();
    SmallVector<llvm::Value *, 8> args;
    args.reserve(llvm::size(func.args()));
    for (auto &indexedArg : llvm::enumerate(func.args())) {
      llvm::Value *argIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), APInt(64, indexedArg.index()));
      llvm::Value *argPtrPtr = builder.CreateGEP(argList, argIndex);
      llvm::Value *argPtr = builder.CreateLoad(argPtrPtr);
      argPtr = builder.CreateBitCast(
          argPtr, indexedArg.value().getType()->getPointerTo());
      llvm::Value *arg = builder.CreateLoad(argPtr);
      args.push_back(arg);
    }

    // Call the implementation function with the extracted arguments.
    llvm::Value *result = builder.CreateCall(&func, args);

    // Assuming the result is one value, potentially of type `void`.
    if (!result->getType()->isVoidTy()) {
      llvm::Value *retIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), APInt(64, llvm::size(func.args())));
      llvm::Value *retPtrPtr = builder.CreateGEP(argList, retIndex);
      llvm::Value *retPtr = builder.CreateLoad(retPtrPtr);
      retPtr = builder.CreateBitCast(retPtr, result->getType()->getPointerTo());
      builder.CreateStore(result, retPtr);
    }

    // The interface function returns void.
    builder.CreateRetVoid();
  }
}

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

  auto llvmModule = translateModuleToLLVMIR(*program->module);
  if (!llvmModule) {
    throw std::runtime_error("could not convert to LLVM IR");
  }

  packFunctionArguments(llvmModule.get());

  std::string error;
  engine = std::unique_ptr<llvm::ExecutionEngine>(
      llvm::EngineBuilder(std::move(llvmModule))
          .setErrorStr(&error)
          .setOptLevel(llvm::CodeGenOpt::Aggressive)
          .setEngineKind(llvm::EngineKind::JIT)
          .create());
  if (!engine) {
    throw std::runtime_error("Failed to create ExecutionEngine: " + error);
  }

  engine->finalizeObject();

  uint64_t addr =
      engine->getFunctionAddress(makePackedFunctionName(program->entry));
  if (!addr) {
    throw std::runtime_error("getFunctionAddress failed");
  }
  jitEntry = reinterpret_cast<Function>(addr);

  descriptors.reserve(bufptrs.size());
  for (unsigned i = 0; i < bufptrs.size(); i++) {
    descriptors.emplace_back(bufptrs[i], program->arguments[i].shape);
    ptrs[i] = descriptors[i].ptr();
  }
}

Executable::~Executable() = default;

void Executable::invoke() { jitEntry(ptrs.data()); }

} // namespace pmlc::compiler
