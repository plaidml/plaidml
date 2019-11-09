// Copyright 2019 Intel Corporation.

#include "plaidml2/op/ffi.h"

#include <mutex>

#include <boost/format.hpp>

#include "plaidml2/core/internal.h"
#include "plaidml2/op/lib/ops.h"

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

plaidml_expr* plaidml_op_make(  //
    plaidml_error* err,         //
    const char* op_name,        //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    Value value{expr};
    auto op = lib::OperationRegistry::Instance()->Resolve(op_name);
    if (!op) {
      throw std::runtime_error(str(boost::format("Operation not registered: %1%") % op_name));
    }
    auto ret = op(value);
#ifdef PLAIDML_AST
    return new plaidml_expr{ret.as_ptr()->expr};
#endif
#ifdef PLAIDML_MLIR
    return new plaidml_expr{ret.as_ptr()->value};
#endif
  });
}

}  // extern "C"
