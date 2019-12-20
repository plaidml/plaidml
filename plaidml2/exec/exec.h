// Copyright 2019 Intel Corporation.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "plaidml2/core/core.h"
#include "plaidml2/edsl/edsl.h"
#include "plaidml2/exec/ffi.h"

namespace plaidml2 {
namespace exec {

inline void init() {
  plaidml::init();
  ffi::call_void(plaidml_exec_init);
}

/// @cond IMPL
namespace details {

struct Deleter {
  void operator()(plaidml_executable* ptr) { ffi::call_void(plaidml_executable_free, ptr); }
  void operator()(plaidml_strings* ptr) { ffi::call_void(plaidml_strings_free, ptr); }
};

template <typename T>
inline std::shared_ptr<T> make_ptr(T* ptr) {
  return std::shared_ptr<T>(ptr, Deleter{});
}

}  // namespace details
/// @endcond IMPL

struct Binding {
  edsl::Tensor tensor;
  Buffer buffer;
};

inline std::vector<std::string> list_devices() {
  auto strs = details::make_ptr(ffi::call<plaidml_strings*>(plaidml_devices_get));
  std::vector<std::string> ret(strs->nstrs);
  for (size_t i = 0; i < ret.size(); i++) {
    ret[i] = ffi::str(strs->strs[i]);
  }
  return ret;
}

inline std::vector<std::string> list_targets() {
  auto strs = details::make_ptr(ffi::call<plaidml_strings*>(plaidml_targets_get));
  std::vector<std::string> ret(strs->nstrs);
  for (size_t i = 0; i < ret.size(); i++) {
    ret[i] = ffi::str(strs->strs[i]);
  }
  return ret;
}

class Executable {
 public:
  Executable(const edsl::Program& program,        //
             const std::vector<Binding>& inputs,  //
             const std::vector<Binding>& outputs)
      : Executable(                           //
            program,                          //
            Settings::get("PLAIDML_DEVICE"),  //
            Settings::get("PLAIDML_TARGET"),  //
            inputs,                           //
            outputs)                          //
  {}

  Executable(const edsl::Program& program,        //
             const std::string& device,           //
             const std::string& target,           //
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
    ptr_ = details::make_ptr(            //
        ffi::call<plaidml_executable*>(  //
            plaidml_compile,             //
            program.as_ptr(),            //
            device.c_str(),              //
            target.c_str(),              //
            raw_inputs.size(),           //
            raw_inputs.data(),           //
            raw_outputs.size(),          //
            raw_outputs.data()));
  }

  void run() {  //
    ffi::call_void(plaidml_executable_run, ptr_.get());
  }

 private:
  std::shared_ptr<plaidml_executable> ptr_;
};

class Binder {
 private:
  using BindingMap = std::map<edsl::TensorRef, Buffer>;

 public:
  explicit Binder(const edsl::Program& program)
      : program_(program),  //
        device_(Settings::get("PLAIDML_DEVICE")),
        target_(Settings::get("PLAIDML_TARGET")) {
    for (const auto& arg : program.inputs()) {
      if (arg.buffer) {
        inputs_[arg.tensor] = *arg.buffer;
      }
    }
    for (const auto& arg : program.outputs()) {
      if (arg.buffer) {
        outputs_[arg.tensor] = *arg.buffer;
      }
    }
  }

  Binder& set_device(const std::string& value) {
    device_ = value;
    return *this;
  }

  Binder& set_target(const std::string& value) {
    target_ = value;
    return *this;
  }

  Buffer input(const edsl::Tensor& tensor) {  //
    return inputs_.at(tensor);
  }

  Buffer output(const edsl::Tensor& tensor) {  //
    return outputs_.at(tensor);
  }

  Binder& set_input(const edsl::Tensor& tensor, const Buffer& buffer) {
    inputs_[tensor] = buffer;
    return *this;
  }

  Binder& set_output(const edsl::Tensor& tensor, const Buffer& buffer) {
    outputs_[tensor] = buffer;
    return *this;
  }

  std::shared_ptr<Executable> compile() {
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    for (const auto& arg : program_.args()) {
      if (arg.is_input) {
        input_bindings.emplace_back(Binding{arg.tensor, get_buffer(&inputs_, arg)});
      } else {
        output_bindings.emplace_back(Binding{arg.tensor, get_buffer(&outputs_, arg)});
      }
    }
    return std::make_shared<Executable>(program_, device_, target_, input_bindings, output_bindings);
  }

 private:
  Buffer get_buffer(BindingMap* map, const edsl::ProgramArgument& arg) {
    auto it = map->find(arg.tensor);
    if (it != map->end()) {
      return it->second;
    }
    TensorShape shape(arg.shape.dtype(), arg.shape.int_dims());
    Buffer buffer{device_, shape};
    map->emplace(arg.tensor, buffer);
    return buffer;
  }

 private:
  edsl::Program program_;
  std::string device_;
  std::string target_;
  std::map<edsl::TensorRef, Buffer> inputs_;
  std::map<edsl::TensorRef, Buffer> outputs_;
};

}  // namespace exec

}  // namespace plaidml2
