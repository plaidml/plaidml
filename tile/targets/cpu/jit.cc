// Copyright 2018, Intel Corp.

#include "tile/targets/cpu/jit.h"

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <algorithm>
#include <deque>
#include <memory>
#include <utility>

#include <half.hpp>

#include "base/util/lookup.h"
#include "tile/stripe/stripe.h"
#include "tile/targets/cpu/compiler.h"
#include "tile/targets/cpu/executable.h"
#include "tile/targets/cpu/link_names.h"

namespace vertexai {
namespace tile {
namespace targets {
namespace cpu {

struct Native::Impl {
  llvm::LLVMContext context;
  ProgramModule module;
  std::unique_ptr<Executable> executable;

  void compile(const stripe::Block& program, const Config& config) {
    Compiler compiler(&context, config);
    module = compiler.CompileProgram(program);
    assert(module.module);
    executable.reset(new Executable(module));
  }

  void run(const std::map<std::string, void*>& buffers) { executable->Run(buffers); }

  void save(const std::string& filename) {
    std::error_code ec;
    llvm::ToolOutputFile result(filename, ec, llvm::sys::fs::F_None);
    WriteBitcodeToFile(*module.module, result.os());
    result.keep();
  }

  void set_perf_attrs(stripe::Block* program) { executable->SetPerfAttrs(program); }
};

Native::Native() : m_impl(new Native::Impl) {}
Native::~Native() {}
void Native::compile(const stripe::Block& program, const Config& config) { m_impl->compile(program, config); }
void Native::run(const std::map<std::string, void*>& buffers) { m_impl->run(buffers); }
void Native::save(const std::string& filename) { m_impl->save(filename); }
void Native::set_perf_attrs(stripe::Block* program) { m_impl->set_perf_attrs(program); }

void JitExecute(const stripe::Block& program, const std::map<std::string, void*>& buffers) {
  Config config;
  JitExecute(program, config, buffers);
}

void JitExecute(const stripe::Block& program, const Config& config, const std::map<std::string, void*>& buffers) {
  llvm::LLVMContext context;
  Compiler compiler(&context, config);
  auto module = compiler.CompileProgram(program);
  Executable executable(std::move(module));
  executable.Run(buffers);
}

void JitProfile(stripe::Block* program, const std::map<std::string, void*>& buffers) {
  llvm::LLVMContext context;
  Config config;
  config.profile_block_execution = true;
  Compiler compiler(&context, config);
  auto module = compiler.CompileProgram(*program);
  Executable executable(std::move(module));
  executable.Run(buffers);
  executable.SetPerfAttrs(program);
}

}  // namespace cpu
}  // namespace targets
}  // namespace tile
}  // namespace vertexai
