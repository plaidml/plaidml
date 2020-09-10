// Copyright 2019 Intel Corporation

#pragma once

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "plaidml/core/ffi.h"
#include "pmlc/dialect/tile/builder.h"

extern "C" {

struct plaidml_string {
  std::string str;
};

struct plaidml_shape {
  mlir::MemRefType type;
};

struct plaidml_dim_expr {
  mlir::Value value;
};

struct plaidml_expr {
  mlir::Value value;
  std::shared_ptr<pmlc::compiler::Program> program;
};

struct plaidml_program {
  std::shared_ptr<pmlc::compiler::Program> program;
};

struct plaidml_buffer {
  std::shared_ptr<pmlc::util::Buffer> buffer;
};

struct plaidml_view {
  std::shared_ptr<pmlc::util::View> view;
};

struct VariantHolder;
using VariantPtr = std::shared_ptr<VariantHolder>;
using Tuple = std::vector<VariantPtr>;

using Variant = std::variant<  //
    std::monostate,            // PLAIDML_VALUE_NONE
    plaidml_dim_expr,          // PLAIDML_VALUE_DIM
    plaidml_expr,              // PLAIDML_VALUE_EXPR
    double,                    // PLAIDML_VALUE_FLOAT
    int64_t,                   // PLAIDML_VALUE_INT
    std::string,               // PLAIDML_VALUE_STR
    Tuple                      // PLAIDML_VALUE_TUPLE
    >;

struct VariantHolder {
  explicit VariantHolder(const Variant& inner);
  Variant inner;
};

struct plaidml_value {
  Variant variant;
};

}  // extern "C"

namespace plaidml::core {

struct GlobalContext {
  static pmlc::dialect::tile::TileBuilder* get() {
    static thread_local pmlc::dialect::tile::TileBuilder builder;
    return &builder;
  }
};

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

plaidml_datatype convertIntoDataType(mlir::Type type);
mlir::Type convertFromDataType(plaidml_datatype dtype, mlir::MLIRContext* context);

// Converts the supplied RAII strings vector into a plaidml_strings*
// for returning via FFI.
plaidml_strings* toFFI(std::vector<std::string> strings);

}  // namespace plaidml::core
