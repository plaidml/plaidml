// Copyright 2019 Intel Corporation.

#include "plaidml2/edsl/ffi.h"

// #include <mutex>
// #include <sstream>

// #include <boost/format.hpp>

#include "base/util/logging.h"
// #include "plaidml2/core/internal.h"
// #include "plaidml2/edsl/derivs.h"
// #include "tile/lang/ast/ast.h"
// #include "tile/lang/ast/gradient.h"

// using plaidml::core::ffi_wrap;
// using plaidml::core::ffi_wrap_void;

extern "C" {

struct plaidml_string {
  std::string str;
};

struct plaidml_expr {
  int foo;
};

}  // extern "C"

namespace {

template <typename T, typename F>
T ffi_wrap(plaidml_error* err, T val, F fn) {
  try {
    err->code = 0;
    err->msg = nullptr;
    return fn();
  } catch (const std::exception& ex) {
    err->code = 1;
    err->msg = new plaidml_string{ex.what()};
    return val;
  } catch (...) {
    err->code = 1;
    err->msg = new plaidml_string{"C++ exception"};
    return val;
  }
}

template <typename F>
void ffi_wrap_void(plaidml_error* err, F fn) {
  try {
    err->code = 0;
    err->msg = nullptr;
    fn();
  } catch (const std::exception& ex) {
    err->code = 1;
    err->msg = new plaidml_string{ex.what()};
  } catch (...) {
    err->code = 1;
    err->msg = new plaidml_string{"C++ exception"};
  }
}

}  // namespace

extern "C" {

void plaidml_edsl_init(  //
    plaidml_error* err) {
  static std::once_flag is_initialized;
  ffi_wrap_void(err, [&] {
    std::call_once(is_initialized, []() {
      IVLOG(1, "plaidml_edsl_init");
      //   plaidml::edsl::deriv::RegisterDerivs();
    });
  });
}

plaidml_expr* plaidml_expr_call(  //
    plaidml_error* err,           //
    const char* fn,               //
    size_t nargs,                 //
    plaidml_expr** args) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    // std::vector<ExprPtr> exprs(nargs);
    // for (size_t i = 0; i < nargs; i++) {
    //   if (!args[i]) {
    //     throw std::runtime_error(str(boost::format("Undefined tensor in call to %1%()") % fn));
    //   }
    //   exprs[i] = args[i]->expr;
    // }
    // return new plaidml_expr{MakeCall(fn, exprs)};
    throw std::runtime_error("TODO");
    return new plaidml_expr{0};
  });
}

}  // extern "C"
