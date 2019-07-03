// Copyright 2019 Intel Corporation.

#include "plaidml2/op/ffi.h"

#include <boost/format.hpp>

#include "plaidml2/core/internal.h"
#include "plaidml2/op/ops.h"

using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using namespace plaidml::edsl;  // NOLINT

extern "C" {

plaidml_expr* plaidml_op_make(  //
    plaidml_error* err,         //
    const char* op_name,        //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    Value value{expr};
    auto op = plaidml::op::OperationRegistry::Instance()->Resolve(op_name);
    if (!op) {
      throw std::runtime_error(str(boost::format("Operation not registered: %1%") % op_name));
    }
    auto ret = op(value);
    return new plaidml_expr{ret.as_ptr()->expr};
  });
}

}  // extern "C"
