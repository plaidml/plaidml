// Copyright 2019 Intel Corporation

#pragma once

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "plaidml/core/ffi.h"
#include "pmlc/ast/ast.h"
#include "pmlc/compiler/program.h"

extern "C" {

struct plaidml_buffer {
  pmlc::util::BufferPtr buffer;
};

struct plaidml_program {
  std::shared_ptr<pmlc::compiler::Program> program;
};

struct plaidml_shape {
  pmlc::util::TensorShape shape;
};

struct plaidml_string {
  std::string str;
};

struct plaidml_value {
  pmlc::ast::VarNodePtr node;
};

}  // extern "C"

namespace plaidml::core {

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

template <typename ResultT, typename ElementT, typename T>
ResultT* ffi_vector(const std::vector<T>& vec) {
  std::vector<std::unique_ptr<ElementT>> ptrs(vec.size());
  std::unique_ptr<ElementT* []> elts { new ElementT*[vec.size()] };
  auto result = std::make_unique<ResultT>();
  for (size_t i = 0; i < vec.size(); i++) {
    ptrs[i].reset(new ElementT{vec[i]});
    elts[i] = ptrs[i].release();
  }
  result->size = vec.size();
  result->elts = elts.release();
  return result.release();
}

template <typename ResultT>
ResultT* ffi_vector(const std::vector<int64_t>& vec) {
  std::unique_ptr<int64_t[]> elts{new int64_t[vec.size()]};
  auto result = std::make_unique<ResultT>();
  for (size_t i = 0; i < vec.size(); i++) {
    elts[i] = vec[i];
  }
  result->size = vec.size();
  result->elts = elts.release();
  return result.release();
}

plaidml_datatype convertIntoDataType(pmlc::util::DataType type);
pmlc::util::DataType convertFromDataType(plaidml_datatype dtype);

}  // namespace plaidml::core
