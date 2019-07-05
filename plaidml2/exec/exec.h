// Copyright 2019 Intel Corporation.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "plaidml2/core/core.h"
#include "plaidml2/edsl/edsl.h"
#include "plaidml2/exec/ffi.h"

namespace plaidml {
namespace exec {

inline void init() {  //
  plaidml::init();
  ffi::call_void(plaidml_exec_init);
}

namespace details {

template <typename T>
struct Deleter {
  std::function<void(plaidml_error*, T*)> fn;
  void operator()(T* ptr) { ffi::call_void(fn, ptr); }
};

inline std::shared_ptr<plaidml_buffer> make_plaidml_buffer(plaidml_buffer* ptr) {
  return std::shared_ptr<plaidml_buffer>(ptr, Deleter<plaidml_buffer>{plaidml_buffer_free});
}

inline std::shared_ptr<plaidml_view> make_plaidml_view(plaidml_view* ptr) {
  return std::shared_ptr<plaidml_view>(ptr, Deleter<plaidml_view>{plaidml_view_free});
}

inline std::shared_ptr<plaidml_executable> make_plaidml_executable(plaidml_executable* ptr) {
  return std::shared_ptr<plaidml_executable>(ptr, Deleter<plaidml_executable>{plaidml_executable_free});
}

}  // namespace details

class View {
  friend class Buffer;

 public:
  char* data() {  //
    return ffi::call<char*>(plaidml_view_data, ptr_.get());
  }

  size_t size() {  //
    return ffi::call<size_t>(plaidml_view_size, ptr_.get());
  }

  void writeback() {  //
    ffi::call_void(plaidml_view_writeback, ptr_.get());
  }

 private:
  explicit View(const std::shared_ptr<plaidml_view>& ptr) : ptr_(ptr) {}

 private:
  std::shared_ptr<plaidml_view> ptr_;
};

class Buffer {
 public:
  Buffer() = default;
  Buffer(const std::string& device_id, const TensorShape& shape)
      : ptr_(details::make_plaidml_buffer(
            ffi::call<plaidml_buffer*>(plaidml_buffer_alloc, device_id.c_str(), shape.nbytes()))),
        shape_(shape) {}

  plaidml_buffer* as_ptr() const {  //
    return ptr_.get();
  }

  View mmap_current() {
    return View(details::make_plaidml_view(ffi::call<plaidml_view*>(plaidml_buffer_mmap_current, ptr_.get())));
  }

  View mmap_discard() {
    return View(details::make_plaidml_view(ffi::call<plaidml_view*>(plaidml_buffer_mmap_discard, ptr_.get())));
  }

  void copy_into(void* dst) {
    auto view = mmap_current();
    memcpy(dst, view.data(), view.size());
  }

  void copy_from(const void* src) {
    auto view = mmap_discard();
    memcpy(view.data(), src, view.size());
    view.writeback();
  }

 private:
  std::shared_ptr<plaidml_buffer> ptr_;
  TensorShape shape_;
};

struct Binding {
  edsl::Tensor tensor;
  Buffer buffer;
};

class Executable {
 public:
  Executable(const edsl::Program& program,        //
             const std::string& device_id,        //
             const std::string& target_id,        //
             const std::vector<Binding>& inputs,  //
             const std::vector<Binding>& outputs) {
    std::vector<plaidml_binding> inputs_storage(inputs.size());
    std::vector<plaidml_binding*> raw_inputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      inputs_storage[i].expr = inputs[i].tensor.as_ptr();
      inputs_storage[i].buffer = inputs[i].buffer.as_ptr();
      raw_inputs[i] = &inputs_storage[i];
    }
    std::vector<plaidml_binding> outputs_storage(outputs.size());
    std::vector<plaidml_binding*> raw_outputs(outputs.size());
    for (size_t i = 0; i < raw_outputs.size(); i++) {
      outputs_storage[i].expr = outputs[i].tensor.as_ptr();
      outputs_storage[i].buffer = outputs[i].buffer.as_ptr();
      raw_outputs[i] = &outputs_storage[i];
    }
    ptr_ = details::make_plaidml_executable(  //
        ffi::call<plaidml_executable*>(       //
            plaidml_compile,                  //
            program.as_ptr(),                 //
            device_id.c_str(),                //
            target_id.c_str(),                //
            raw_inputs.size(),                //
            raw_inputs.data(),                //
            raw_outputs.size(),               //
            raw_outputs.data()));
  }

  void run() {  //
    ffi::call_void(plaidml_executable_run, ptr_.get());
  }

 private:
  std::shared_ptr<plaidml_executable> ptr_;
};

inline std::vector<std::string> list_devices() {
  auto count = ffi::call<size_t>(plaidml_device_list_count);
  std::vector<plaidml_string*> strs(count);
  ffi::call_void(plaidml_device_list, strs.size(), strs.data());
  std::vector<std::string> ret(count);
  for (size_t i = 0; i < ret.size(); i++) {
    ret[i] = ffi::str(strs[i]);
  }
  return ret;
}

inline std::vector<std::string> list_targets() {
  auto count = ffi::call<size_t>(plaidml_target_list_count);
  std::vector<plaidml_string*> strs(count);
  ffi::call_void(plaidml_target_list, strs.size(), strs.data());
  std::vector<std::string> ret(count);
  for (size_t i = 0; i < ret.size(); i++) {
    ret[i] = ffi::str(strs[i]);
  }
  return ret;
}

}  // namespace exec

}  // namespace plaidml
