// Copyright 2019 Intel Corporation.

#pragma once

#include <string>
#include <vector>

#include "plaidml2/edsl/edsl.h"
#include "plaidml2/op/ffi.h"

namespace plaidml {
namespace op {

static const char* const NCX = "ncx";
static const char* const NXC = "nxc";
static const char* const KCX = "kcx";
static const char* const XCK = "xck";

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
    const edsl::Tensor& I_or_O,              //
    const edsl::Tensor& F_or_O,              //
    const std::vector<int>& strides,         //
    const std::vector<int>& dilations,       //
    const std::vector<int>& data_dilations,  //
    const std::vector<int>& filter_shape,    //
    int groups,                              //
    const std::string& autopad_mode,         //
    const std::vector<int>& manual_padding,  //
    const std::string& input_layout,         //
    const std::string& filter_layout,        //
    const std::string& group_layout,         //
    bool winograd_allowed,                   //
    const std::string& name,                 //
    const std::string& autogroup_mode,       //
    const std::string& deriv_mode,           //
    const std::vector<int>& result_shape     //
) {
  auto args = edsl::make_tuple(          //
      I_or_O,                            //
      F_or_O,                            //
      edsl::make_tuple(strides),         //
      edsl::make_tuple(dilations),       //
      edsl::make_tuple(data_dilations),  //
      edsl::make_tuple(filter_shape),    //
      groups,                            //
      autopad_mode,                      //
      edsl::make_tuple(manual_padding),  //
      input_layout,                      //
      filter_layout,                     //
      group_layout,                      //
      winograd_allowed,                  //
      name,                              //
      autogroup_mode,                    //
      deriv_mode,                        //
      edsl::make_tuple(result_shape));
  return details::op("convolution", args).as_tensor();
}

inline edsl::Tensor mean(const edsl::Tensor& I, const edsl::Value& axes = edsl::None(), bool keepdims = false) {
  auto args = edsl::make_tuple(I, axes, keepdims);
  return details::op("mean", args).as_tensor();
}

inline edsl::Tensor reshape(const edsl::Tensor& I, const edsl::Value& dims) {
  auto args = edsl::make_tuple(I, dims);
  return details::op("reshape", args).as_tensor();
}

inline edsl::Tensor square(const edsl::Tensor& x) {  //
  return details::op("square", edsl::Value(x)).as_tensor();
}

inline edsl::Tensor spatial_padding(  //
    const edsl::Tensor& x,            //
    const std::vector<int>& lo_pads,  //
    const std::vector<int>& hi_pads,  //
    const std::string& data_layout) {
  auto args = edsl::make_tuple(x, edsl::make_tuple(lo_pads), edsl::make_tuple(hi_pads), data_layout);
  return details::op("spatial_padding", edsl::Value(args)).as_tensor();
}

inline edsl::Tensor sum(const edsl::Tensor& I, const edsl::Value& axes = edsl::None(), bool keepdims = false) {
  auto args = edsl::make_tuple(I, axes, keepdims);
  return details::op("sum", args).as_tensor();
}

inline edsl::Tensor variance(const edsl::Tensor& I, const edsl::Value& axes = edsl::None(), bool keepdims = false) {
  auto args = edsl::make_tuple(I, axes, keepdims);
  return details::op("variance", args).as_tensor();
}

}  // namespace op
}  // namespace plaidml
