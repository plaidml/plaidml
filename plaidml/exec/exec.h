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
  Executable(const Program& program,             //
             const std::vector<Buffer>& inputs,  //
             const std::vector<Buffer>& outputs)
      : Executable(program, "", inputs, outputs) {}

  ///
  /// Executable constructor
  ///
  Executable(const Program& program,             //
             const std::string& device,          //
             const std::vector<Buffer>& inputs,  //
             const std::vector<Buffer>& outputs)
      : inputs_(inputs), outputs_(outputs) {
    std::vector<plaidml_buffer*> raw_inputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      raw_inputs[i] = inputs[i].as_ptr();
    }
    std::vector<plaidml_buffer*> raw_outputs(outputs.size());
    for (size_t i = 0; i < raw_outputs.size(); i++) {
      raw_outputs[i] = outputs[i].as_ptr();
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
  void run() { ffi::call_void(plaidml_executable_run, ptr_.get()); }

 private:
  std::shared_ptr<plaidml_executable> ptr_;
  std::vector<Buffer> inputs_;
  std::vector<Buffer> outputs_;
};

}  // namespace exec

}  // namespace plaidml
