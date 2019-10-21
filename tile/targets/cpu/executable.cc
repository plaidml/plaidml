// Copyright 2019, Intel Corp.

#include "tile/targets/cpu/executable.h"

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
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

#include "base/util/env.h"
#include "base/util/lookup.h"
#include "tile/stripe/stripe.h"
#include "tile/targets/cpu/link_names.h"

#if defined(_WIN32)
// As of 2019-08-01, libxsmm doesn't compile on Windows if UNICODE is defined, since it passes
// an ANSI string to CreateMutexW().  So we rewrite it.
#undef CreateMutex
#define CreateMutex CreateMutexA
#endif

// libxsmm
#include "libxsmm_source.h"  // NOLINT

namespace vertexai {
namespace tile {
namespace targets {
namespace cpu {

class Runtime : public llvm::LegacyJITSymbolResolver {
 public:
  explicit Runtime(const std::map<std::string, void*> externals) : externals_(externals) {}
  llvm::JITSymbol findSymbol(const std::string&) override;
  llvm::JITSymbol findSymbolInLogicalDylib(const std::string&) override;

 private:
  std::map<std::string, void*> externals_;
};

Executable::Executable(const ProgramModule& module) : parameters_(module.parameters) {
  std::string errStr;
  std::unique_ptr<llvm::LegacyJITSymbolResolver> rez(new Runtime(module.externals));
  assert(module.module);
  std::unique_ptr<llvm::Module> clone(llvm::CloneModule(*module.module));
  auto ee = llvm::EngineBuilder(std::move(clone))
                .setErrorStr(&errStr)
                .setEngineKind(llvm::EngineKind::JIT)
                .setVerifyModules(true)
                .setSymbolResolver(std::move(rez))
                .create();
  if (ee) {
    if (env::Get("VTUNE_PROFILE") == "1") {
      ee->RegisterJITEventListener(llvm::JITEventListener::createIntelJITEventListener());
    }
    ee->finalizeObject();
    engine_.reset(ee);
  } else {
    throw std::runtime_error("Failed to create ExecutionEngine: " + errStr);
  }
}

void Executable::Run(const std::map<std::string, void*>& buffers) {
  std::vector<void*> args(parameters_.size());
  for (size_t i = 0; i < args.size(); ++i) {
    args[i] = safe_at(buffers, parameters_[i]);
  }
  void* argvec = args.data();
  uint64_t entrypoint = engine_->getFunctionAddress(invoker_name_);
  ((void (*)(void*))entrypoint)(argvec);
}

void Executable::SetPerfAttrs(stripe::Block* block) {
  // Look up the performance counters for this block.
  // Apply their values as tags.
  std::string count_name = profile_count_name_ + block->name;
  uint64_t count_addr = engine_->getGlobalValueAddress(count_name);
  if (count_addr) {
    block->set_attr("execution_count", *reinterpret_cast<int64_t*>(count_addr));
  }
  std::string ticks_name = profile_ticks_name_ + block->name;
  uint64_t ticks_addr = engine_->getGlobalValueAddress(ticks_name);
  if (ticks_addr) {
    block->set_attr("execution_ticks", *reinterpret_cast<int64_t*>(ticks_addr));
  }
  std::string loop_body_name = profile_loop_body_name_ + block->name;
  uint64_t loop_ticks_addr = engine_->getGlobalValueAddress(loop_body_name);
  if (loop_ticks_addr) {
    block->set_attr("loop_body_ticks", *reinterpret_cast<int64_t*>(loop_ticks_addr));
  }
  // Recurse through nested blocks.
  for (const auto& stmt : block->stmts) {
    if (stmt->kind() == stripe::StmtKind::Block) {
      SetPerfAttrs(stripe::Block::Downcast(stmt).get());
    }
  }
}

namespace rt {
// Implementations of support functions the tile backend will link against,
// that we won't be able to resolve from system libraries.
float h2f(half_float::half n) { return n; }
half_float::half f2h(float n) { return half_float::half_cast<half_float::half>(n); }
void prng_step(uint32_t* in_state, uint32_t* out_state, float* buf, size_t count) {
  // A reimplementation of the PRNG from tile/lang/gen_special.cc.
  // x_n = (s1_n ^ s2_n ^ s3_n)
  // s1_{n+1} = (((s1_n & 4294967294) <<12) ^ (((s1_n <<13) ^ s1_n) >>19))
  // s2_{n+1} = (((s2_n & 4294967288) << 4) ^ (((s2_n << 2) ^ s2_n) >>25))
  // s3_{n+1} = (((s3_n & 4294967280) <<17) ^ (((s3_n << 3) ^ s3_n) >>11))
  for (size_t i = 0; i < count; ++i) {
    buf[i] = (in_state[0] ^ in_state[1] ^ in_state[2]) / 4294967296.0;
    out_state[0] = (((in_state[0] & 4294967294) << 12) ^ (((in_state[0] << 13) ^ in_state[0]) >> 19));
    out_state[1] = (((in_state[1] & 4294967288) << 4) ^ (((in_state[1] << 2) ^ in_state[1]) >> 25));
    out_state[2] = (((in_state[2] & 4294967280) << 17) ^ (((in_state[2] << 3) ^ in_state[2]) >> 11));
    in_state = out_state;
  }
}

void RunTimeLogEntry(char* str, char* extra, float address) {
  IVLOG(1, "RunTimeLogEntry: " << str << ":" << extra << ":" /* 0x" << std::hex */ << address);
}

typedef void (*libxsmm_function)(const void* a, const void* b, void* c);
void XSMMRTCaller(libxsmm_function func, const void* aPtr, const void* bPtr, void* cPtr) { func(aPtr, bPtr, cPtr); }

}  // namespace rt

template <typename T>
llvm::JITEvaluatedSymbol symInfo(T ptr) {
  auto flags = llvm::JITSymbolFlags::None;
  auto addr = reinterpret_cast<uintptr_t>(ptr);
  return llvm::JITEvaluatedSymbol(addr, flags);
}

llvm::JITSymbol Runtime::findSymbol(const std::string& name) {
  static std::map<std::string, llvm::JITEvaluatedSymbol> symbols{
      {"__gnu_h2f_ieee", symInfo(rt::h2f)},
      {"__gnu_f2h_ieee", symInfo(rt::f2h)},
      {"___extendhfsf2", symInfo(rt::h2f)},
      {"___truncsfhf2", symInfo(rt::f2h)},
      {"_libxsmm_dmmdispatch", symInfo(libxsmm_dmmdispatch)},
      {"_libxsmm_smmdispatch", symInfo(libxsmm_smmdispatch)},
      {"_libxsmm_wimmdispatch", symInfo(libxsmm_wimmdispatch)},
      {"_prng_step", symInfo(rt::prng_step)},
      {"_RunTimeLogEntry", symInfo(rt::RunTimeLogEntry)},  // For debugging
      {"_XSMMRTCaller", symInfo(rt::XSMMRTCaller)},
      {"libxsmm_dmmdispatch", symInfo(libxsmm_dmmdispatch)},
      {"libxsmm_smmdispatch", symInfo(libxsmm_smmdispatch)},
      {"libxsmm_wimmdispatch", symInfo(libxsmm_wimmdispatch)},
      {"prng_step", symInfo(rt::prng_step)},
      {"RunTimeLogEntry", symInfo(rt::RunTimeLogEntry)},  // For debugging
      {"XSMMRTCaller", symInfo(rt::XSMMRTCaller)},
  };
  auto loc_rt = symbols.find(name);
  if (loc_rt != symbols.end()) {
    return loc_rt->second;
  }
  auto loc_extern = externals_.find(name);
  if (loc_extern != externals_.end()) {
    return symInfo(loc_extern->second);
  }
  if (name.size() > 1 && name[0] == '_') {
    loc_extern = externals_.find(name.substr(1));
    if (loc_extern != externals_.end()) {
      return symInfo(loc_extern->second);
    }
  }
  auto ptr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(name);
  // If we failed to resolve the symbol, and its first character is an
  // underscore, try again without the underscore. The code may have been
  // generated for a system whose loader expects every symbol to have an
  // underscore prefix, but the DynamicLibrary module expects no prefix.
  if (!ptr && name[0] == '_' && name.size() > 1) {
    ptr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(name.substr(1));
  }
  if (ptr) {
    auto info = symInfo(ptr);
    symbols.emplace(name, info);
    return info;
  }
  throw std::runtime_error("failed to resolve external symbol reference: \"" + name + "\"");
}

llvm::JITSymbol Runtime::findSymbolInLogicalDylib(const std::string& name) { return llvm::JITSymbol(nullptr); }

}  // namespace cpu
}  // namespace targets
}  // namespace tile
}  // namespace vertexai
