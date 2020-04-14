// Copyright 2019 Intel Corporation.

#pragma once

#include <string>
#include <vector>

#include "plaidml/edsl/edsl.h"
#include "plaidml/op/ffi.h"

namespace plaidml {
namespace op {

inline void init() {
  plaidml::init();
  plaidml::edsl::init();
  ffi::call_void(plaidml_op_init);
}

namespace details {

inline edsl::Value op(const std::string& name, const edsl::Value& args) {
  return edsl::Value(ffi::call<plaidml_value*>(plaidml_op_make, name.c_str(), args.as_ptr()));
}

}  // namespace details

static const int AUTO_DIM_MATCH = 0;
static const int AUTO_DIM_FILL = -1;

enum class AutoGroupMode {
  UNGROUPED,  // Group size explicitly 1
  EXPLICIT,   // Group size explicitly specified, > 1
  AUTO,       // Group size determined from shapes of I and F
  DEPTHWISE,  // for channelized convolutions (i.e. where G = CI)
  _LAST,
};

enum class AutoPadMode {
  NONE,
  NOTSET = NONE,
  EXPLICIT = NONE,
  SAME_LOWER,
  SAME_UPPER,
  VALID,
  _LAST,
};

enum class ConvDerivMode {
  NONE,    // Forward Pass
  DATA,    // Computing derivative of input data (or equivalently a transposed conv)
  FILTER,  // Computing derivative of filters
  _LAST,
};

// For grouped convolutions, in the filters (i.e. weights/kernel) tensor, there
// are multiple ways of laying out the channels. For a convolution with:
//  G groups
//  C input channels
//  K output channels
// there must be a total of (C * K) / G channel combinations. This is generally
// accomplished by having one of the input or output channel dimensions include
// the group and having the other be the within-group channel; but the group
// can also be included as a separate dimension. This gives the following total
// sizes for the channel dimensions:
//  SEPARATE: G, C/G, K/G
//  IN_C:     C, K/G
//  IN_K:     C/G, K
// SEPARATE is the layout with the group given as a separate dimension. IN_C is
// the layout with the group included in C, and with the K dim representing the
// within-group output channel. IN_K is the layout with the group included in K
// with the C dim representing the within-group input channel.
// The NONE layout is used for convolutions that aren't grouped.
enum class GroupLayout {
  NONE,      // Not grouped
  SEPARATE,  // Group given as a separate dimension
  IN_C,      // Group included in the input channels dimension
  IN_K,      // Group included in the output channels dimensiono
  _LAST,
};

enum class InterpolationMode {
  NEAREST,
  BILINEAR,
  _LAST,
};

enum class PoolMode {
  AVG,
  MAX,
  MIN,
  SUM,
  _LAST,
};

enum class TensorLayout {
  NXC,
  NCX,
  KCX,
  XCK,
  GKCX,
  XGCK,
  _LAST,
};

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

inline edsl::Tensor binary_crossentropy(const edsl::Tensor& I, const edsl::Tensor& O, double epsilon) {
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

class convolution {
 public:
  explicit convolution(edsl::Tensor I, edsl::Tensor F) : I_(I), F_(F) {}

  convolution& strides(const std::vector<int>& strides) {
    strides_ = strides;
    return *this;
  }

  convolution& dilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
    return *this;
  }

  convolution& data_dilations(const std::vector<int>& data_dilations) {
    data_dilations_ = data_dilations;
    return *this;
  }

  convolution& filter_shape(const std::vector<int>& filter_shape) {
    filter_shape_ = filter_shape;
    return *this;
  }

  convolution& groups(int groups) {
    groups_ = groups;
    return *this;
  }

  convolution& manual_padding(const std::vector<int>& manual_padding) {
    manual_padding_ = manual_padding;
    return *this;
  }

  convolution& autopad_mode(AutoPadMode autopad_mode) {
    autopad_mode_ = autopad_mode;
    return *this;
  }

  convolution& input_layout(TensorLayout input_layout) {
    input_layout_ = input_layout;
    return *this;
  }

  convolution& filter_layout(TensorLayout filter_layout) {
    filter_layout_ = filter_layout;
    return *this;
  }

  convolution& group_layout(GroupLayout group_layout) {
    group_layout_ = group_layout;
    return *this;
  }

  convolution& winograd_allowed(bool winograd_allowed) {
    winograd_allowed_ = winograd_allowed;
    return *this;
  }

  convolution& name(const std::string& name) {
    name_ = name;
    return *this;
  }

  convolution& autogroup_mode(AutoGroupMode autogroup_mode) {
    autogroup_mode_ = autogroup_mode;
    return *this;
  }

  convolution& deriv_mode(ConvDerivMode deriv_mode) {
    deriv_mode_ = deriv_mode;
    return *this;
  }

  convolution& result_shape(const std::vector<int>& result_shape) {
    result_shape_ = result_shape;
    return *this;
  }

  operator edsl::Tensor() const {
    auto args = edsl::make_tuple(           //
        I_,                                 //
        F_,                                 //
        edsl::make_tuple(strides_),         //
        edsl::make_tuple(dilations_),       //
        edsl::make_tuple(data_dilations_),  //
        edsl::make_tuple(filter_shape_),    //
        groups_,                            //
        static_cast<int>(autopad_mode_),    //
        edsl::make_tuple(manual_padding_),  //
        static_cast<int>(input_layout_),    //
        static_cast<int>(filter_layout_),   //
        static_cast<int>(group_layout_),    //
        winograd_allowed_,                  //
        name_,                              //
        static_cast<int>(autogroup_mode_),  //
        static_cast<int>(deriv_mode_),      //
        edsl::make_tuple(result_shape_));
    return details::op("convolution", args).as_tensor();
  }

 private:
  edsl::Tensor I_;
  edsl::Tensor F_;
  std::vector<int> strides_;
  std::vector<int> dilations_;
  std::vector<int> data_dilations_;
  std::vector<int> filter_shape_;
  int groups_ = 1;
  std::vector<int> manual_padding_;
  AutoPadMode autopad_mode_;
  TensorLayout input_layout_;
  TensorLayout filter_layout_;
  GroupLayout group_layout_ = GroupLayout::NONE;
  bool winograd_allowed_ = false;
  std::string name_;
  AutoGroupMode autogroup_mode_ = AutoGroupMode::UNGROUPED;
  ConvDerivMode deriv_mode_ = ConvDerivMode::NONE;
  std::vector<int> result_shape_;
};

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

inline edsl::Tensor elu(const edsl::Tensor& I, double alpha) {
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

inline edsl::Tensor hard_sigmoid(const edsl::Tensor& I, double slope) {
  auto args = edsl::make_tuple(I, slope);
  return details::op("hard_sigmoid", args).as_tensor();
}

inline edsl::Tensor image_resize(const edsl::Tensor& I, const std::vector<int>& factors,
                                 InterpolationMode interpolation, TensorLayout layout) {
  auto args = edsl::make_tuple(I, edsl::make_tuple(factors), static_cast<int>(interpolation), static_cast<int>(layout));
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
    PoolMode pool_mode,                      //
    const std::vector<int>& pool_size,       //
    const std::vector<int>& strides,         //
    AutoPadMode autopad_mode,                //
    const std::vector<int>& manual_padding,  //
    TensorLayout input_layout,               //
    bool include_padding_in_avg = false,     //
    bool use_ceil_for_output_shape = false   //
) {
  auto args = edsl::make_tuple(          //
      I,                                 //
      static_cast<int>(pool_mode),       //
      edsl::make_tuple(pool_size),       //
      edsl::make_tuple(strides),         //
      static_cast<int>(autopad_mode),    //
      edsl::make_tuple(manual_padding),  //
      static_cast<int>(input_layout),    //
      include_padding_in_avg,            //
      use_ceil_for_output_shape);
  return details::op("pool", args).as_tensor();
}

inline edsl::Tensor prod(const edsl::Tensor& I, const edsl::Value& axes = edsl::None(), bool keepdims = false) {
  auto args = edsl::make_tuple(I, axes, keepdims);
  return details::op("prod", args).as_tensor();
}

class relu {
 public:
  explicit relu(const edsl::Tensor& I) : I_(I) {}

  relu& alpha(const edsl::Tensor& alpha) {
    alpha_ = alpha;
    return *this;
  }

  relu& max_value(const edsl::Tensor& max_value) {
    max_value_ = max_value;
    return *this;
  }

  relu& threshold(double threshold) {
    threshold_ = threshold;
    return *this;
  }

  operator edsl::Tensor() const {
    auto args = edsl::make_tuple(I_, alpha_, max_value_, threshold_);
    return details::op("relu", args).as_tensor();
  }

 private:
  edsl::Tensor I_;
  edsl::Tensor alpha_;
  edsl::Tensor max_value_;
  double threshold_ = 0.0;
};

inline edsl::Tensor reorg_yolo(const edsl::Tensor& I, int stride, bool decrease) {
  auto args = edsl::make_tuple(I, stride, decrease);
  return details::op("reorg_yolo", args).as_tensor();
}

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
    TensorLayout data_layout) {
  auto args = edsl::make_tuple(x, edsl::make_tuple(lo_pads), edsl::make_tuple(hi_pads), static_cast<int>(data_layout));
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
