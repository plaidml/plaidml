// Copyright 2019 Intel Corporation.

#include "plaidml2/exec/ffi.h"

#include <mutex>

#include <boost/format.hpp>

#include "plaidml2/core/internal.h"
#include "tile/targets/targets.h"

using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using plaidml::core::GetPlatform;
using vertexai::context::Context;
using vertexai::tile::Allocator;
using vertexai::tile::Buffer;
using vertexai::tile::ConstBufferManager;
using vertexai::tile::Program;
using vertexai::tile::View;
using vertexai::tile::lang::ast::ParamExpr;
using vertexai::tile::targets::GetConfigs;

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

}  // namespace

extern "C" {

struct plaidml_executable {
  using BufferMap = std::map<std::string, std::shared_ptr<Buffer>>;
  BufferMap input_bufs;
  BufferMap output_bufs;
  std::shared_ptr<Program> program;
};

void plaidml_exec_init(  //
    plaidml_error* err) {
  static std::once_flag is_initialized;
  ffi_wrap_void(err, [&] {
    std::call_once(is_initialized, []() {  //
      IVLOG(1, "plaidml_exec_init");
      GetPlatform();
    });
  });
}

size_t plaidml_device_list_count(  //
    plaidml_error* err) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return GetPlatform()->ListDevices().size();
  });
}

void plaidml_device_list(  //
    plaidml_error* err,    //
    size_t ndevices,       //
    plaidml_string** device_ids) {
  ffi_wrap_void(err, [&] {
    auto devices = GetPlatform()->ListDevices();
    for (size_t i = 0; i < std::min(ndevices, devices.size()); i++) {
      device_ids[i] = new plaidml_string{devices[i]};
    }
  });
}

plaidml_executable* plaidml_compile(  //
    plaidml_error* err,               //
    plaidml_program* program,         //
    const char* device_id,            //
    const char* target,               //
    size_t ninputs,                   //
    plaidml_binding** inputs,         //
    size_t noutputs,                  //
    plaidml_binding** outputs) {
  return ffi_wrap<plaidml_executable*>(err, nullptr, [&] {
    auto configs = GetConfigs().configs();
    if (!configs.count(target)) {
      throw std::runtime_error(str(boost::format("Unknown target specified: %1%") % target));
    }
    Context ctx;
    ConstBufferManager const_bufs;
    const_bufs.allocator = std::make_shared<PlatformAllocator>(device_id);
    auto exec = new plaidml_executable{};
    exec->program = GetPlatform()->MakeProgram(ctx, device_id, target, program->eval.runinfo, &const_bufs);
    for (size_t i = 0; i < ninputs; i++) {
      auto param_expr = std::dynamic_pointer_cast<ParamExpr>(inputs[i]->expr->expr);
      if (!param_expr) {
        throw std::runtime_error("Buffers can only be bound to ParamExprs");
      }
      param_expr->buffer = inputs[i]->buffer->buffer;
    }
    const auto& program_inputs = program->eval.inputs;
    for (size_t i = 0; i < program_inputs.size(); i++) {
      const auto& name = program->eval.runinfo.program.inputs[i].name;
      if (!program_inputs[i]->buffer) {
        throw std::runtime_error(str(boost::format("Unbound buffer for input: %1%") % name));
      }
      exec->input_bufs[name] = program_inputs[i]->buffer;
    }
    const auto& program_outputs = program->eval.outputs;
    for (size_t i = 0; i < noutputs; i++) {
      if (!outputs[i] || !outputs[i]->expr || !outputs[i]->buffer) {
        throw std::runtime_error("Undefined output bindings");
      }
      auto expr = outputs[i]->expr->expr;
      auto it = std::find(program_outputs.begin(), program_outputs.end(), expr);
      size_t j = std::distance(program_outputs.begin(), it);
      const auto& name = program->eval.runinfo.program.outputs[j];
      exec->output_bufs[name] = outputs[i]->buffer->buffer;
    }
    return exec;
  });
}

size_t plaidml_target_list_count(  //
    plaidml_error* err) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return GetConfigs().configs().size();
  });
}

void plaidml_target_list(  //
    plaidml_error* err,    //
    size_t ntargets,       //
    plaidml_string** targets) {
  ffi_wrap_void(err, [&] {
    auto configs = GetConfigs().configs();
    size_t i = 0;
    for (const auto& kvp : configs) {
      if (i >= ntargets) {
        break;
      }
      targets[i++] = new plaidml_string{kvp.first};
    }
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
    Context ctx;
    exec->program->Run(ctx, exec->input_bufs, exec->output_bufs).get();
  });
}

}  // extern "C"
