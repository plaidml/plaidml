// Copyright 2019 Intel Corporation.

#pragma once

#include <memory>
#include <string>

#include "plaidml2/core/ffi.h"
#include "tile/base/platform.h"

#ifdef PLAIDML_AST
#include "tile/base/shape.h"
#include "tile/lang/ast/ast.h"
#endif
#ifdef PLAIDML_MLIR
#include "pmlc/dialect/stripe/types.h"
#include "pmlc/dialect/tile/builder.h"
#endif

extern "C" {

struct plaidml_string {
  std::string str;
};

struct plaidml_shape {
#ifdef PLAIDML_AST
  vertexai::tile::TensorShape shape;
#endif
#ifdef PLAIDML_MLIR
  pmlc::dialect::stripe::TensorType type;
#endif
};

struct plaidml_expr {
#ifdef PLAIDML_AST
  vertexai::tile::lang::ast::ExprPtr expr;
#endif
#ifdef PLAIDML_MLIR
  mlir::Value* value = nullptr;
#endif
};

struct plaidml_program {
#ifdef PLAIDML_AST
  vertexai::tile::lang::ast::ProgramEvaluation eval;
#endif
#ifdef PLAIDML_MLIR
  std::shared_ptr<pmlc::dialect::tile::TileProgram> program;
#endif
};

struct plaidml_buffer {
  std::shared_ptr<vertexai::tile::Buffer> buffer;
};

struct plaidml_view {
  std::shared_ptr<vertexai::tile::View> view;
};

}  // extern "C"

namespace plaidml::core {

struct GlobalContext {
#ifdef PLAIDML_MLIR
  static pmlc::dialect::tile::TileBuilder* get() {
    static thread_local pmlc::dialect::tile::TileBuilder builder;
    return &builder;
  }
#endif

  static vertexai::context::Context* getContext() {
    static vertexai::context::Context context;
    return &context;
  }
};

struct PlatformHolder {
  PlatformHolder();
  std::unique_ptr<vertexai::tile::Platform> platform;
  vertexai::tile::Platform* operator->() { return platform.get(); }
};

PlatformHolder& GetPlatform();

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

}  // namespace plaidml::core
