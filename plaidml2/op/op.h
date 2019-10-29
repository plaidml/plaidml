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

inline edsl::Tensor abs(const edsl::Tensor& I) {
  auto args = edsl::make_tuple(I);
  return details::op("abs", args).as_tensor();
}

inline edsl::Tensor all(const edsl::Tensor& I, const edsl::Value& axes = edsl::None(), bool keepdims = false) {
  auto args = edsl::make_tuple(I, axes, keepdims);
  return details::op("all", args).as_tensor();
}

inline edsl::Tensor any(const edsl::Tensor& I, const edsl::Value& axes = edsl::None(), bool keepdims = false) {
  auto args = edsl::make_tuple(I, axes, keepdims);
  return details::op("any", args).as_tensor();
}

inline edsl::Tensor argmax(const edsl::Tensor& I, const edsl::Value& axes = edsl::None()) {
  auto args = edsl::make_tuple(I, axes);
  return details::op("argmax", args).as_tensor();
}

inline edsl::Tensor binary_crossentropy(const edsl::Tensor& I, const edsl::Tensor& O, float epsilon) {
  auto args = edsl::make_tuple(I, O, epsilon);
  return details::op("binary_crossentropy", args).as_tensor();
}

inline edsl::Tensor clip(const edsl::Tensor& I, const edsl::Tensor& min, const edsl::Tensor& max) {
  auto args = edsl::make_tuple(I, min, max);
  return details::op("clip", args).as_tensor();
}

inline edsl::Tensor concatenate(const std::vector<edsl::Tensor>& tensors, int axis) {
  auto args = edsl::make_tuple(edsl::make_tuple(tensors), axis);
  return details::op("concatenate", args).as_tensor();
}

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

inline edsl::Tensor cumprod(const edsl::Tensor& I, int axis) {
  auto args = edsl::make_tuple(I, axis);
  return details::op("cumprod", args).as_tensor();
}

inline edsl::Tensor cumsum(const edsl::Tensor& I, int axis) {
  auto args = edsl::make_tuple(I, axis);
  return details::op("cumsum", args).as_tensor();
}

inline edsl::Tensor dot(const edsl::Tensor& I, const edsl::Tensor& K) {
  auto args = edsl::make_tuple(I, K);
  return details::op("dot", args).as_tensor();
}

inline edsl::Tensor elu(const edsl::Tensor& I, float alpha) {
  auto args = edsl::make_tuple(I, alpha);
  return details::op("elu", args).as_tensor();
}

inline edsl::Tensor expand_dims(const edsl::Tensor& I, int axis) {
  auto args = edsl::make_tuple(I, axis);
  return details::op("expand_dims", args).as_tensor();
}

inline edsl::Tensor flip(const edsl::Tensor& I, int axis) {
  auto args = edsl::make_tuple(I, axis);
  return details::op("flip", args).as_tensor();
}

inline edsl::Tensor hard_sigmoid(const edsl::Tensor& I, float slope) {
  auto args = edsl::make_tuple(I, slope);
  return details::op("hard_sigmoid", args).as_tensor();
}

inline edsl::Tensor image_resize(const edsl::Tensor& I, const std::vector<int>& factors,
                                 const std::string& interpolation, const std::string& layout) {
  auto args = edsl::make_tuple(I, edsl::make_tuple(factors), interpolation, layout);
  return details::op("image_resize", args).as_tensor();
}

inline edsl::Tensor max(const edsl::Tensor& I,  // NOLINT(build/include_what_you_use)
                        const edsl::Value& axes = edsl::None(), bool keepdims = false) {
  auto args = edsl::make_tuple(I, axes, keepdims);
  return details::op("max", args).as_tensor();
}

inline edsl::Tensor maximum(const edsl::Tensor& X, const edsl::Tensor& Y) {
  auto args = edsl::make_tuple(X, Y);
  return details::op("maximum", args).as_tensor();
}

inline edsl::Tensor mean(const edsl::Tensor& I, const edsl::Value& axes = edsl::None(), bool keepdims = false) {
  auto args = edsl::make_tuple(I, axes, keepdims);
  return details::op("mean", args).as_tensor();
}

inline edsl::Tensor min(const edsl::Tensor& I, const edsl::Value& axes = edsl::None(),  // NOLINT
                        bool keepdims = false) {
  auto args = edsl::make_tuple(I, axes, keepdims);
  return details::op("min", args).as_tensor();
}

inline edsl::Tensor minimum(const edsl::Tensor& X, const edsl::Tensor& Y) {
  auto args = edsl::make_tuple(X, Y);
  return details::op("minimum", args).as_tensor();
}

inline edsl::Tensor pool(                    //
    const edsl::Tensor I,                    //
    const std::string& pool_mode,            //
    const std::vector<int>& pool_size,       //
    const std::vector<int>& strides,         //
    const std::string& autopad_mode,         //
    const std::vector<int>& manual_padding,  //
    const std::string& input_layout,         //
    bool include_padding_in_avg = false,     //
    bool use_ceil_for_output_shape = false   //
) {
  auto args = edsl::make_tuple(          //
      I,                                 //
      pool_mode,                         //
      edsl::make_tuple(pool_size),       //
      edsl::make_tuple(strides),         //
      autopad_mode,                      //
      edsl::make_tuple(manual_padding),  //
      input_layout,                      //
      include_padding_in_avg,            //
      use_ceil_for_output_shape);
  return details::op("pool", args).as_tensor();
}

inline edsl::Tensor prod(const edsl::Tensor& I, const edsl::Value& axes = edsl::None(), bool keepdims = false) {
  auto args = edsl::make_tuple(I, axes, keepdims);
  return details::op("prod", args).as_tensor();
}

class relu {
 protected:
  edsl::Tensor I_;
  edsl::Tensor alpha_;
  edsl::Tensor max_value_;
  float threshold_ = 0.0;

 public:
  explicit relu(const edsl::Tensor& I) : I_(I) {}

  relu alpha(const edsl::Tensor& alpha) {
    alpha_ = alpha;
    return *this;
  }

  relu max_value(const edsl::Tensor& max_value) {
    max_value_ = max_value;
    return *this;
  }

  relu threshold(float threshold) {
    threshold_ = threshold;
    return *this;
  }

  operator edsl::Tensor() const {
    // actually make the relu from the op lib
    auto args = edsl::make_tuple(I_, alpha_, max_value_, threshold_);
    return details::op("relu", args).as_tensor();
  }
};

inline edsl::Tensor repeat(const edsl::Tensor& I, int repeats, int axis) {
  auto args = edsl::make_tuple(I, repeats, axis);
  return details::op("repeat", args).as_tensor();
}

inline edsl::Tensor reshape(const edsl::Tensor& I, const edsl::Value& dims) {
  auto args = edsl::make_tuple(I, dims);
  return details::op("reshape", args).as_tensor();
}

inline edsl::Tensor sigmoid(const edsl::Tensor& I) {
  auto args = edsl::make_tuple(I);
  return details::op("sigmoid", args).as_tensor();
}

inline edsl::Tensor slice(const edsl::Tensor& I, const std::vector<int>& slices) {
  auto args = edsl::make_tuple(I, edsl::make_tuple(slices));
  return details::op("slice", args).as_tensor();
}

inline edsl::Tensor softmax(const edsl::Tensor& I, int axis) {
  auto args = edsl::make_tuple(I, axis);
  return details::op("softmax", args).as_tensor();
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

inline edsl::Tensor squeeze(const edsl::Tensor& I, const std::vector<int>& axes) {
  auto args = edsl::make_tuple(I, edsl::make_tuple(axes));
  return details::op("squeeze", args).as_tensor();
}

inline edsl::Tensor sum(const edsl::Tensor& I, const edsl::Value& axes = edsl::None(), bool keepdims = false) {
  auto args = edsl::make_tuple(I, axes, keepdims);
  return details::op("sum", args).as_tensor();
}

inline edsl::Tensor tile(const edsl::Tensor& I, const std::vector<int>& tiling_factors) {
  auto args = edsl::make_tuple(I, edsl::make_tuple(tiling_factors));
  return details::op("tile", args).as_tensor();
}

inline edsl::Tensor transpose(const edsl::Tensor& I, const edsl::Value& axes = edsl::None()) {
  auto args = edsl::make_tuple(I, axes);
  return details::op("transpose", args).as_tensor();
}

inline edsl::Tensor variance(const edsl::Tensor& I, const edsl::Value& axes = edsl::None(), bool keepdims = false) {
  auto args = edsl::make_tuple(I, axes, keepdims);
  return details::op("variance", args).as_tensor();
}

}  // namespace op
}  // namespace plaidml
