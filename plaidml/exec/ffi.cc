// Copyright 2019 Intel Corporation.

#include "plaidml/exec/ffi.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/Support/FormatVariadic.h"

#include "plaidml/core/internal.h"
#include "pmlc/compiler/executable.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using pmlc::compiler::EngineKind;
using pmlc::compiler::Executable;
using pmlc::compiler::ProgramArgument;
using pmlc::util::Buffer;
using pmlc::util::BufferPtr;
using namespace mlir;  // NOLINT[build/namespaces]

namespace {

const char* kCpuDevice = "llvm_cpu.0";

std::vector<ProgramArgument> BindProgramArguments(  //
    plaidml_program* program,                       //
    size_t ninputs,                                 //
    plaidml_binding** inputs,                       //
    size_t noutputs,                                //
    plaidml_binding** outputs) {
  llvm::DenseMap<Value, BufferPtr> input_bindings;
  for (unsigned i = 0; i < ninputs; i++) {
    input_bindings[inputs[i]->expr->value] = inputs[i]->buffer->buffer;
  }
  llvm::DenseMap<Value, BufferPtr> output_bindings;
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
    args[i] = std::move(arg);
  }
  return args;
}

}  // namespace

extern "C" {

struct plaidml_executable {
  std::unique_ptr<Executable> exec;
};

void plaidml_exec_init(  //
    plaidml_error* err) {
  ffi_wrap_void(err, [&] {  //
    IVLOG(1, "plaidml_exec_init");
  });
}

plaidml_strings* plaidml_devices_get(  //
    plaidml_error* err) {
  return ffi_wrap<plaidml_strings*>(err, nullptr, [&] {
    std::vector<std::string> devices{kCpuDevice};
    auto strs = new plaidml_string*[devices.size()];
    for (size_t i = 0; i < devices.size(); i++) {
      strs[i] = new plaidml_string{devices[i]};
    }
    return new plaidml_strings{devices.size(), strs};
  });
}

plaidml_executable* plaidml_jit(  //
    plaidml_error* err,           //
    plaidml_program* program,     //
    const char* device,           //
    size_t ninputs,               //
    plaidml_binding** inputs,     //
    size_t noutputs,              //
    plaidml_binding** outputs) {
  return ffi_wrap<plaidml_executable*>(err, nullptr, [&] {
    IVLOG(1, "JITing for device: " << device);
    auto args = BindProgramArguments(program, ninputs, inputs, noutputs, outputs);
    auto exec = std::make_unique<plaidml_executable>();
    std::vector<void*> bufptrs(args.size());
    for (unsigned i = 0; i < args.size(); i++) {
      auto view = args[i].buffer->MapCurrent();
      bufptrs[i] = view->data();
    }
    EngineKind kind = EngineKind::MCJIT;
    auto jit = pmlc::util::getEnvVar("LLVM_JIT");
    if (jit == "ORC") {
      kind = EngineKind::OrcJIT;
    } else if (jit == "MCJIT") {
      kind = EngineKind::MCJIT;
    }
    exec->exec = std::make_unique<Executable>(program->program, bufptrs, kind);
    return exec.release();
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
  ffi_wrap_void(err, [&] {  //
    exec->exec->invoke();
  });
}

}  // extern "C"
