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

inline void init() {
  plaidml::init();
  ffi::call_void(plaidml_exec_init);
}

namespace details {

template <typename T>
struct Deleter {
  std::function<void(plaidml_error*, T*)> fn;
  void operator()(T* ptr) { ffi::call_void(fn, ptr); }
};

inline std::shared_ptr<plaidml_executable> make_plaidml_executable(plaidml_executable* ptr) {
  return std::shared_ptr<plaidml_executable>(ptr, Deleter<plaidml_executable>{plaidml_executable_free});
}

}  // namespace details

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

  static std::shared_ptr<Executable> compile(const edsl::Program& program, const std::vector<edsl::Tensor>& inputs) {
    auto device = Settings::get("PLAIDML_DEVICE");
    auto target = Settings::get("PLAIDML_TARGET");

    std::vector<Binding> input_bindings;
    for (auto input : inputs) {
      auto shape = input.shape();
      TensorShape tensor_shape(shape.dtype(), shape.int_dims());
      input_bindings.emplace_back(Binding{input, Buffer{device, tensor_shape}});
    }

    std::vector<Binding> output_bindings;
    for (auto output : program.outputs()) {
      auto shape = output.shape();
      TensorShape tensor_shape(shape.dtype(), shape.int_dims());
      output_bindings.emplace_back(Binding{output, Buffer{device, tensor_shape}});
    }

    return std::make_shared<Executable>(program, device, target, input_bindings, output_bindings);
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
