// Copyright 2019 Intel Corporation.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "plaidml/core/core.h"
#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/ffi.h"

namespace plaidml {
namespace exec {

///
/// Initializes the PlaidML Execution API.
///
inline void init() {
  plaidml::init();
  ffi::call_void(plaidml_exec_init);
}

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

///
/// \defgroup exec_objects Objects
///

///
/// \ingroup exec_objects
/// \struct Binding
/// Bindings bind a Tensor to a Buffer.
///
struct Binding {
  /// The tensor to bind.
  edsl::Tensor tensor;
  /// The buffer to be bound to.
  Buffer buffer;
};

///
/// \defgroup exec_functions Functions
///

///
/// \ingroup exec_functions
/// Lists the available devices.
/// Use this list of devices to set the environment variable `PLAIDML_DEVICE`.
/// \return vector<string>
///
inline std::vector<std::string> list_devices() {
  auto strs = details::make_ptr(ffi::call<plaidml_strings*>(plaidml_devices_get));
  std::vector<std::string> ret(strs->size);
  for (size_t i = 0; i < ret.size(); i++) {
    ret[i] = ffi::str(strs->elts[i]);
  }
  return ret;
}

///
/// \ingroup exec_objects
/// \class Executable
/// This is an Executable.
///
class Executable {
 public:
  ///
  /// Executable constructor
  ///
  Executable(const edsl::Program& program,        //
             const std::vector<Binding>& inputs,  //
             const std::vector<Binding>& outputs)
      : Executable(                           //
            program,                          //
            Settings::get("PLAIDML_DEVICE"),  //
            inputs,                           //
            outputs)                          //
  {}

  ///
  /// Executable constructor
  ///
  Executable(const edsl::Program& program,        //
             const std::string& device,           //
             const std::vector<Binding>& inputs,  //
             const std::vector<Binding>& outputs) {
    std::vector<plaidml_binding> inputs_storage(inputs.size());
    std::vector<plaidml_binding*> raw_inputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      inputs_storage[i].expr = ffi::call<plaidml_expr*>(plaidml_expr_clone, inputs[i].tensor.as_ptr());
      inputs_storage[i].buffer = ffi::call<plaidml_buffer*>(plaidml_buffer_clone, inputs[i].buffer.as_ptr());
      raw_inputs[i] = &inputs_storage[i];
    }
    std::vector<plaidml_binding> outputs_storage(outputs.size());
    std::vector<plaidml_binding*> raw_outputs(outputs.size());
    for (size_t i = 0; i < raw_outputs.size(); i++) {
      outputs_storage[i].expr = ffi::call<plaidml_expr*>(plaidml_expr_clone, outputs[i].tensor.as_ptr());
      outputs_storage[i].buffer = ffi::call<plaidml_buffer*>(plaidml_buffer_clone, outputs[i].buffer.as_ptr());
      raw_outputs[i] = &outputs_storage[i];
    }
    ptr_ = details::make_ptr(            //
        ffi::call<plaidml_executable*>(  //
            plaidml_jit,                 //
            program.as_ptr(),            //
            device.c_str(),              //
            raw_inputs.size(),           //
            raw_inputs.data(),           //
            raw_outputs.size(),          //
            raw_outputs.data()));
  }

  ///
  /// run
  ///
  void run() {  //
    ffi::call_void(plaidml_executable_run, ptr_.get());
  }

 private:
  std::shared_ptr<plaidml_executable> ptr_;
};

///
/// \ingroup exec_objects
/// \struct Binder
/// This is a Binder.
///
class Binder {
 private:
  using BindingMap = std::map<edsl::TensorRef, Buffer>;

 public:
  ///
  /// Constructs a Binder.
  ///
  explicit Binder(const edsl::Program& program)
      : program_(program),  //
        device_(Settings::get("PLAIDML_DEVICE")) {
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

  ///
  /// Set the device for the Binder to use.
  /// \param value string
  /// \return Binder
  ///
  Binder& set_device(const std::string& value) {
    device_ = value;
    return *this;
  }

  ///
  /// input
  /// \param tensor Tensor
  /// \return Buffer
  ///
  Buffer input(const edsl::Tensor& tensor) {  //
    return inputs_.at(tensor);
  }

  ///
  /// output
  /// \param tensor Tensor
  /// \return Buffer
  ///
  Buffer output(const edsl::Tensor& tensor) {  //
    return outputs_.at(tensor);
  }

  ///
  /// set_input
  /// \param tensor Tensor
  /// \param buffer Buffer
  /// \return Binder
  ///
  Binder& set_input(const edsl::Tensor& tensor, const Buffer& buffer) {
    inputs_[tensor] = buffer;
    return *this;
  }

  ///
  /// set_output
  /// \param tensor Tensor
  /// \param buffer Buffer
  /// \return Binder
  ///
  Binder& set_output(const edsl::Tensor& tensor, const Buffer& buffer) {
    outputs_[tensor] = buffer;
    return *this;
  }

  ///
  /// compile
  /// \return shared_ptr<Executable>
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
    return std::make_shared<Executable>(program_, device_, input_bindings, output_bindings);
  }

 private:
  Buffer get_buffer(BindingMap* map, const edsl::ProgramArgument& arg) {
    auto it = map->find(arg.tensor);
    if (it != map->end()) {
      return it->second;
    }
    TensorShape shape(arg.shape.dtype(), arg.shape.sizes());
    Buffer buffer{device_, shape};
    map->emplace(arg.tensor, buffer);
    return buffer;
  }

 private:
  edsl::Program program_;
  std::string device_;
  std::map<edsl::TensorRef, Buffer> inputs_;
  std::map<edsl::TensorRef, Buffer> outputs_;
};

}  // namespace exec

}  // namespace plaidml
