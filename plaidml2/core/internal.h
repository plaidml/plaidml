// Copyright 2019 Intel Corporation.

#pragma once

#include <memory>
#include <string>

#include "plaidml2/core/ffi.h"
#include "pmlc/dialect/stripe/types.h"
#include "pmlc/dialect/tile/builder.h"
#include "tile/base/platform.h"
#include "tile/base/shape.h"

extern "C" {

struct plaidml_string {
  std::string str;
};

struct plaidml_shape {
  pmlc::dialect::stripe::TensorType type;
};

struct plaidml_expr {
  mlir::Value* value = nullptr;
};

struct plaidml_program {
  std::shared_ptr<pmlc::dialect::tile::TileProgram> program;
};

struct plaidml_buffer {
  std::shared_ptr<vertexai::tile::Buffer> buffer;
};

struct plaidml_view {
  std::shared_ptr<vertexai::tile::View> view;
};

}  // extern "C"

namespace plaidml::core {

using pmlc::dialect::tile::TileBuilder;

struct GlobalContext {
  static TileBuilder* get() {
    static thread_local TileBuilder builder;
    return &builder;
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
