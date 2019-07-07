// Copyright 2019 Intel Corporation.

#pragma once

#include <string>
#include <vector>

#include "plaidml2/edsl/edsl.h"
#include "plaidml2/op/ffi.h"

namespace plaidml {
namespace op {

static const char* const NCHW = "NCHW";
static const char* const NHWC = "NHWC";
static const char* const KCHW = "KCHW";
static const char* const HWCK = "HWCK";

inline void init() {  //
  plaidml::init();
  plaidml::edsl::init();
  ffi::call_void(plaidml_op_init);
}

namespace details {

inline edsl::Value op(const std::string& name, const edsl::Value& args) {
  return edsl::Value(ffi::call<plaidml_expr*>(plaidml_op_make, name.c_str(), args.as_ptr()));
}

}  // namespace details

inline edsl::Tensor convolution(             //
    const edsl::Tensor& input,               //
    const edsl::Tensor& weights,             //
    const std::vector<int>& stride,          //
    const std::vector<int>& padding,         //
    const std::vector<int>& dilation,        //
    int groups,                              //
    const std::string& input_layout = NHWC,  //
    const std::string& weights_layout = HWCK) {
  auto args = edsl::make_tuple(    //
      input,                       //
      weights,                     //
      edsl::make_tuple(stride),    //
      edsl::make_tuple(padding),   //
      edsl::make_tuple(dilation),  //
      groups,                      //
      input_layout,                //
      weights_layout);
  return details::op("convolution", args).as_tensor();
}

inline edsl::Tensor mean(const edsl::Tensor& I, const edsl::Value& axes = edsl::None(), bool keepdims = false) {
  auto args = edsl::make_tuple(I, axes, keepdims);
  return details::op("mean", args).as_tensor();
}

inline edsl::Tensor square(const edsl::Tensor& x) {  //
  return details::op("square", edsl::Value(x)).as_tensor();
}

inline edsl::Tensor sum(const edsl::Tensor& I, const edsl::Value& axes = edsl::None(), bool keepdims = false) {
  auto args = edsl::make_tuple(I, axes, keepdims);
  return details::op("sum", args).as_tensor();
}

}  // namespace op
}  // namespace plaidml
