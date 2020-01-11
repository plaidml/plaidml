// Copyright 2019 Intel Corporation.

#include "plaidml2/op/ffi.h"

#include <mutex>

#include "llvm/Support/FormatVariadic.h"

#include "plaidml2/core/internal.h"
#include "plaidml2/op/lib/ops.h"
#include "pmlc/util/logging.h"

using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using namespace plaidml::op;    // NOLINT
using namespace plaidml::edsl;  // NOLINT

extern "C" {

void plaidml_op_init(  //
    plaidml_error* err) {
  static std::once_flag is_initialized;
  ffi_wrap_void(err, [&] {
    std::call_once(is_initialized, []() {
      IVLOG(1, "plaidml_op_init");
      lib::RegisterOps();
    });
  });
}

plaidml_value* plaidml_op_make(  //
    plaidml_error* err,          //
    const char* op_name,         //
    plaidml_value* value) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    auto op = lib::OperationRegistry::Instance()->Resolve(op_name);
    if (!op) {
      throw std::runtime_error(llvm::formatv("Operation not registered: {0}", op_name).str());
    }
    auto ret = op(Value{value});
    return new plaidml_value{ret.as_ptr()->variant};
  });
}

}  // extern "C"
