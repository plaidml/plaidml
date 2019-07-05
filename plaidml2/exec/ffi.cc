// Copyright 2019 Intel Corporation.

#include "plaidml2/exec/ffi.h"

#include <mutex>

#include <boost/format.hpp>

#include "plaidml2/core/internal.h"
#include "tile/platform/local_machine/platform.h"
#include "tile/targets/targets.h"

using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using vertexai::context::Context;
using vertexai::tile::Allocator;
using vertexai::tile::Buffer;
using vertexai::tile::ConstBufferManager;
using vertexai::tile::Program;
using vertexai::tile::View;
using vertexai::tile::local_machine::Platform;
using vertexai::tile::targets::GetConfigs;

namespace {

Platform* GetPlatform() {
  static Platform platform;
  return &platform;
}

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

struct plaidml_buffer {
  std::shared_ptr<Buffer> buffer;
};

struct plaidml_view {
  std::shared_ptr<View> view;
};

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
      if (!inputs[i] || !inputs[i]->expr || !inputs[i]->buffer) {
        throw std::runtime_error("Undefined input bindings");
      }
      auto expr = inputs[i]->expr->expr;
      const auto& inputs_ord = program->eval.inputs;
      auto it = std::find(inputs_ord.begin(), inputs_ord.end(), expr.get());
      size_t j = std::distance(inputs_ord.begin(), it);
      const auto& name = program->eval.runinfo.program.inputs[j].name;
      exec->input_bufs[name] = inputs[i]->buffer->buffer;
    }
    for (size_t i = 0; i < noutputs; i++) {
      if (!outputs[i] || !outputs[i]->expr || !outputs[i]->buffer) {
        throw std::runtime_error("Undefined output bindings");
      }
      auto expr = outputs[i]->expr->expr;
      const auto& outputs_ord = program->eval.outputs;
      auto it = std::find(outputs_ord.begin(), outputs_ord.end(), expr.get());
      size_t j = std::distance(outputs_ord.begin(), it);
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

void plaidml_buffer_free(  //
    plaidml_error* err,    //
    plaidml_buffer* buffer) {
  ffi_wrap_void(err, [&] {  //
    delete buffer;
  });
}

plaidml_buffer* plaidml_buffer_alloc(  //
    plaidml_error* err,                //
    const char* device_id,             //
    size_t size) {
  return ffi_wrap<plaidml_buffer*>(err, nullptr, [&] {
    Context ctx;
    auto buffer = GetPlatform()->MakeBuffer(ctx, device_id, size);
    return new plaidml_buffer{buffer};
  });
}

plaidml_view* plaidml_buffer_mmap_current(  //
    plaidml_error* err,                     //
    plaidml_buffer* buffer) {
  return ffi_wrap<plaidml_view*>(err, nullptr, [&] {  //
    Context ctx;
    return new plaidml_view{buffer->buffer->MapCurrent(ctx).get()};
  });
}

plaidml_view* plaidml_buffer_mmap_discard(  //
    plaidml_error* err,                     //
    plaidml_buffer* buffer) {
  return ffi_wrap<plaidml_view*>(err, nullptr, [&] {  //
    Context ctx;
    return new plaidml_view{buffer->buffer->MapDiscard(ctx)};
  });
}

void plaidml_view_free(  //
    plaidml_error* err,  //
    plaidml_view* view) {
  ffi_wrap_void(err, [&] {  //
    delete view;
  });
}

char* plaidml_view_data(  //
    plaidml_error* err,   //
    plaidml_view* view) {
  return ffi_wrap<char*>(err, nullptr, [&] {  //
    return view->view->data();
  });
}

size_t plaidml_view_size(  //
    plaidml_error* err,    //
    plaidml_view* view) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return view->view->size();
  });
}

void plaidml_view_writeback(  //
    plaidml_error* err,       //
    plaidml_view* view) {
  ffi_wrap_void(err, [&] {
    Context ctx;
    view->view->WriteBack(ctx);
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
