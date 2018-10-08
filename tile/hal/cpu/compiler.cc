// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cpu/compiler.h"

#include <llvm/ExecutionEngine/ExecutionEngine.h>

#include <exception>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "base/util/logging.h"
#include "tile/hal/cpu/emitllvm.h"
#include "tile/hal/cpu/executable.h"
#include "tile/hal/cpu/library.h"
#include "tile/hal/cpu/runtime.h"
#include "tile/lang/semprinter.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

Compiler::Compiler() {}

boost::future<std::unique_ptr<hal::Library>> Compiler::Build(const context::Context& ctx,
                                                             const std::vector<lang::KernelInfo>& kernel_info,
                                                             const hal::proto::HardwareSettings&) {
  if (!kernel_info.size()) {
    return boost::make_ready_future(std::unique_ptr<hal::Library>{
        compat::make_unique<cpu::Library>(std::vector<std::shared_ptr<llvm::ExecutionEngine>>{}, kernel_info)});
  }

  static std::once_flag init_once;
  std::call_once(init_once, []() {
    LLVMInitializeNativeTarget();
    LLVMLinkInMCJIT();
    LLVMInitializeNativeAsmPrinter();
    LLVMInitializeNativeAsmParser();
  });
  std::vector<std::shared_ptr<llvm::ExecutionEngine>> engines;
  for (const auto& ki : kernel_info) {
    BuildKernel(ki, &engines);
  }
  std::unique_ptr<hal::Library> lib(new cpu::Library(engines, kernel_info));
  return boost::make_ready_future<>(std::move(lib));
}

void Compiler::BuildKernel(const lang::KernelInfo& ki, std::vector<std::shared_ptr<llvm::ExecutionEngine>>* engines) {
  if (VLOG_IS_ON(4)) {
    sem::Print debug_emit(*ki.kfunc);
    VLOG(4) << "Compiling kernel:\n" << debug_emit.str();
  }

  // Generate LLVM IR for the kernel.
  Emit emit;
  assert(ki.kfunc);
  ki.kfunc->Accept(emit);
  // Generate an invoker function wrapping the kernel params: we will pass in
  // a pointer to an array of buffer pointers, and it will extract the members.
  // This way we can use a single C function pointer invocation for all kernels.
  GenerateInvoker(ki, emit.result().get());
  if (VLOG_IS_ON(4)) {
    VLOG(4) << "Generated IR:\n" << emit.str();
  }
  // Compile the IR into executable code.
  std::string errStr;
  std::unique_ptr<llvm::RuntimeDyld::SymbolResolver> rez(new Runtime);
  llvm::ExecutionEngine* ee = llvm::EngineBuilder(std::move(emit.result()))
                                  .setErrorStr(&errStr)
                                  .setEngineKind(llvm::EngineKind::JIT)
                                  .setVerifyModules(true)
                                  .setSymbolResolver(std::move(rez))
                                  .create();
  if (ee) {
    ee->finalizeObject();
    engines->emplace_back(ee);
  } else {
    std::cerr << "Failed to create ExecutionEngine: " << errStr << std::endl;
  }
}

void Compiler::GenerateInvoker(const lang::KernelInfo& ki, llvm::Module* module) {
  // Generate a wrapper function for this kernel so that we can call it
  // generically no matter how many parameters it expects. The wrapper will
  // accept a pointer to an array of pointers, and it will call the kernel
  // using each element of the array as an argument. Following the array of
  // buffer pointers, it will pass along the GridSize value for the current
  // work index.
  llvm::LLVMContext& context(llvm::getGlobalContext());
  llvm::IRBuilder<> builder(context);
  // LLVM doesn't have the notion of a void pointer, so we'll pretend all of
  // these buffers are arrays of int32.
  llvm::Type* inttype = llvm::Type::getInt32Ty(context);
  llvm::Type* ptrtype = inttype->getPointerTo();
  size_t param_count = ki.kfunc->params.size();
  llvm::Type* arrayptr = ptrtype->getPointerTo();
  unsigned archbits = module->getDataLayout().getPointerSizeInBits();
  llvm::Type* sizetype = llvm::IntegerType::get(context, archbits);
  auto gridSizeCount = std::tuple_size<lang::GridSize>::value;
  llvm::Type* gridSizeType = llvm::ArrayType::get(sizetype, gridSizeCount);
  // The invoker gets two arguments: the first is a pointer to an array of
  // buffer pointers, one for each kernel parameter; the second is the workIndex
  // value, a lang::GridSize array of three indexes. All output is produced by
  // writing to the buffers, so the return type is void.
  std::vector<llvm::Type*> invoker_args{arrayptr, gridSizeType->getPointerTo()};
  llvm::Type* voidtype = llvm::Type::getVoidTy(context);
  auto invokertype = llvm::FunctionType::get(voidtype, invoker_args, false);
  auto linkage = llvm::Function::ExternalLinkage;
  std::string invokername = Executable::InvokerName(ki.kname);
  const char* nstr = invokername.c_str();
  auto invoker = llvm::Function::Create(invokertype, linkage, nstr, module);
  // The invoker has no branches so we'll only need a single basic block.
  auto block = llvm::BasicBlock::Create(context, "block", invoker);
  builder.SetInsertPoint(block);
  // We'll look up the kernel by name and implicitly bitcast it so we can call
  // it using our int32-pointers in place of whatever it actually expects;
  // LLVM will tolerate this mismatch when we use getOrInsertFunction.
  std::vector<llvm::Type*> kernel_args(param_count, ptrtype);
  kernel_args.push_back(gridSizeType->getPointerTo());
  auto kerneltype = llvm::FunctionType::get(voidtype, kernel_args, false);
  auto kernel = module->getOrInsertFunction(ki.kname, kerneltype);
  auto ai = invoker->arg_begin();
  llvm::Value* argvec = &(*ai);
  llvm::Value* workIndex = &(*++ai);
  // The body of the invoker will simply compute the element pointer for each
  // argument value in order, then load the value.
  std::vector<llvm::Value*> args;
  for (unsigned i = 0; i < param_count; ++i) {
    llvm::Value* index = llvm::ConstantInt::get(inttype, i);
    std::vector<llvm::Value*> idxList{index};
    llvm::Value* elptr = builder.CreateGEP(argvec, idxList);
    args.push_back(builder.CreateLoad(elptr));
  }
  args.push_back(workIndex);
  // Having built the argument list, we'll call the actual kernel using the
  // parameter signature it expects.
  builder.CreateCall(kernel, args, "");
  builder.CreateRet(nullptr);
}

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
