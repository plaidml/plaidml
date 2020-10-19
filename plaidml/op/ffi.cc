// Copyright 2019 Intel Corporation.

#include "plaidml/op/ffi.h"

#include <mutex>

#include "llvm/Support/FormatVariadic.h"

#include "plaidml/core/internal.h"
#include "plaidml/op/lib/ops.h"
#include "pmlc/util/logging.h"

using plaidml::core::ffi_wrap;
using namespace plaidml::op;    // NOLINT
using namespace plaidml::edsl;  // NOLINT

extern "C" {

plaidml_value* plaidml_op_make(  //
    plaidml_error* err,          //
    const char* op_name,         //
    plaidml_value* value) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    static std::once_flag is_initialized;
    std::call_once(is_initialized, []() { lib::RegisterOps(); });
    auto op = lib::OperationRegistry::Instance()->Resolve(op_name);
    if (!op) {
      throw std::runtime_error(llvm::formatv("Operation not registered: {0}", op_name).str());
    }
    auto ret = op(Value{value});
    return new plaidml_value{ret.as_ptr()->node};
  });
}

}  // extern "C"
