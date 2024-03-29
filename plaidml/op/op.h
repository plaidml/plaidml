// Copyright 2019 Intel Corporation.

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "plaidml/edsl/edsl.h"
#include "plaidml/op/ffi.h"

namespace plaidml {
namespace op {

inline void init() {
  plaidml::init();
  plaidml::edsl::init();
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
  EXPLICIT,
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

enum class EpsMode {
  ADD,
  MAX,
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

enum class PadMode {
  CONSTANT,
  EDGE,
  REFLECT,
  SYMMETRIC,
  _LAST,
};

enum class BoxesDecodeMode {
  NMS,
  SSD,
  _LAST,
};

enum class TopKSortType {
  VALUE,
  INDEX,
  _LAST,
};

enum class NmsStyle {
  CAFFE,
  MXNET,
  OV,
  _LAST,
};

struct Integers {
  Integers(const std::vector<int>& elts)  // NOLINT[runtime/explicit]
      : value(edsl::make_tuple(elts)) {}
  Integers(const std::vector<int64_t>& elts) {  // NOLINT[runtime/explicit]
    value = edsl::make_tuple(elts);
  }
  Integers(const std::vector<size_t>& elts) {  // NOLINT[runtime/explicit]
    std::vector<int64_t> casted(elts.begin(), elts.end());
    value = edsl::make_tuple(casted);
  }
  Integers(const std::initializer_list<int>& elts)  // NOLINT[runtime/explicit]
      : Integers(std::vector<int>(elts)) {}
  Integers(const std::initializer_list<int64_t>& elts)  // NOLINT[runtime/explicit]
      : Integers(std::vector<int64_t>(elts)) {}
  Integers(const std::initializer_list<size_t>& elts)  // NOLINT[runtime/explicit]
      : Integers(std::vector<size_t>(elts)) {}

  edsl::Value value;
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

inline edsl::Tensor broadcast(const edsl::Tensor& I, const std::vector<int64_t>& result_shape,
                              const std::vector<int64_t>& bcast_axes) {
  auto args = edsl::make_tuple(I, edsl::make_tuple(result_shape), edsl::make_tuple(bcast_axes));
  return details::op("broadcast", args).as_tensor();
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
  explicit convolution(edsl::Tensor I, edsl::Tensor F)
      : I_(I),
        F_(F),
        groups_(1),
        autopad_mode_(AutoPadMode::SAME_UPPER),
        input_layout_(TensorLayout::NXC),
        filter_layout_(TensorLayout::XCK),
        group_layout_(GroupLayout::NONE),
        winograd_allowed_(false),
        autogroup_mode_(AutoGroupMode::UNGROUPED),
        deriv_mode_(ConvDerivMode::NONE),
        infer_result_shape_(false) {}

  convolution& strides(Integers elts) {
    strides_ = elts.value;
    return *this;
  }

  convolution& dilations(Integers elts) {
    dilations_ = elts.value;
    return *this;
  }

  convolution& data_dilations(Integers elts) {
    data_dilations_ = elts.value;
    return *this;
  }

  convolution& filter_shape(Integers elts) {
    filter_shape_ = elts.value;
    return *this;
  }

  convolution& groups(int groups) {
    groups_ = groups;
    return *this;
  }

  convolution& manual_padding(Integers elts) {
    manual_padding_ = elts.value;
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

  convolution& result_shape(Integers elts) {
    result_shape_ = elts.value;
    return *this;
  }

  convolution& infer_result_shape(bool infer_result_shape) {
    infer_result_shape_ = infer_result_shape;
    return *this;
  }

  operator edsl::Tensor() const {
    auto args = edsl::make_tuple(           //
        I_,                                 //
        F_,                                 //
        strides_,                           //
        dilations_,                         //
        data_dilations_,                    //
        filter_shape_,                      //
        groups_,                            //
        static_cast<int>(autopad_mode_),    //
        manual_padding_,                    //
        static_cast<int>(input_layout_),    //
        static_cast<int>(filter_layout_),   //
        static_cast<int>(group_layout_),    //
        winograd_allowed_,                  //
        name_,                              //
        static_cast<int>(autogroup_mode_),  //
        static_cast<int>(deriv_mode_),      //
        result_shape_,                      //
        infer_result_shape_);
    return details::op("convolution", args).as_tensor();
  }

 private:
  edsl::Tensor I_;
  edsl::Tensor F_;
  edsl::Value strides_;           // Default: empty (builds vector of 1s in oplib)
  edsl::Value dilations_;         // Default: empty (builds vector of 1s in oplib)
  edsl::Value data_dilations_;    // Default: empty (builds vector of 1s in oplib)
  edsl::Value filter_shape_;      // Default: empty (i.e. no filter dim check)
  int groups_;                    // Default: 1
  edsl::Value manual_padding_;    // Default: empty (i.e. no manual padding)
  AutoPadMode autopad_mode_;      // Default: AutoPadMode::SAME_UPPER
  TensorLayout input_layout_;     // Default: TensorLayout::NXC
  TensorLayout filter_layout_;    // Default: TensorLayout::XCK
  GroupLayout group_layout_;      // Default: GroupLayout::NONE
  bool winograd_allowed_;         // Default: false
  std::string name_;              // Default: empty (oplib currently renames "" to "conv")
  AutoGroupMode autogroup_mode_;  // Default: AutoGroupMode::UNGROUPED
  ConvDerivMode deriv_mode_;      // Default: ConvDerivMode::NONE
  edsl::Value result_shape_;      // Default: empty (i.e. unspecified)
  bool infer_result_shape_;       // Default: false
};

inline edsl::Tensor cumprod(const edsl::Tensor& I, int axis) {
  auto args = edsl::make_tuple(I, axis);
  return details::op("cumprod", args).as_tensor();
}

inline edsl::Tensor cumsum(const edsl::Tensor& I, int axis, bool exclusive = false) {
  auto args = edsl::make_tuple(I, axis, exclusive);
  return details::op("cumsum", args).as_tensor();
}

inline edsl::Tensor dot(const edsl::Tensor& I, const edsl::Tensor& K, const std::string& name = "") {
  auto args = edsl::make_tuple(I, K, name);
  return details::op("dot", args).as_tensor();
}

inline edsl::Tensor elu(const edsl::Tensor& I, double alpha) {
  auto args = edsl::make_tuple(I, alpha);
  return details::op("elu", args).as_tensor();
}

class explicit_padding {
 public:
  explicit explicit_padding(const edsl::Tensor& I, const std::vector<int>& lo_pads, const std::vector<int>& hi_pads)
      : I_(I), lo_pads_(lo_pads), hi_pads_(hi_pads), mode_(PadMode::CONSTANT), padval_(edsl::Constant(0)) {}

  explicit_padding& lo_pads(const std::vector<int>& lo_pads) {
    lo_pads_ = lo_pads;
    return *this;
  }

  explicit_padding& hi_pads(const std::vector<int>& hi_pads) {
    hi_pads_ = hi_pads;
    return *this;
  }

  explicit_padding& mode(PadMode mode) {
    mode_ = mode;
    return *this;
  }

  explicit_padding& padval(const edsl::Tensor& padval) {
    padval_ = padval;
    return *this;
  }

  operator edsl::Tensor() const {
    auto args =
        edsl::make_tuple(I_, edsl::make_tuple(lo_pads_), edsl::make_tuple(hi_pads_), static_cast<int>(mode_), padval_);
    return details::op("explicit_padding", args).as_tensor();
  }

 private:
  edsl::Tensor I_;
  std::vector<int> lo_pads_;
  std::vector<int> hi_pads_;
  PadMode mode_;
  edsl::Tensor padval_;
};

inline edsl::Tensor flip(const edsl::Tensor& I, int axis) {
  auto args = edsl::make_tuple(I, axis);
  return details::op("flip", args).as_tensor();
}

class gatherND {
 public:
  explicit gatherND(const edsl::Tensor& x, const edsl::Tensor& y) : x_(x), y_(y) {}

  gatherND& interpolationMode(edsl::InterpolationMode mode) {
    interpolation_mode_ = mode;
    return *this;
  }

  gatherND& batchDims(int batch_dims) {
    batch_dims_ = batch_dims;
    return *this;
  }

  operator edsl::Tensor() {
    auto args = edsl::make_tuple(x_, y_, batch_dims_, static_cast<int>(interpolation_mode_));
    return details::op("gatherND", args).as_tensor();
  }

 private:
  edsl::Tensor x_;
  edsl::Tensor y_;
  uint64_t batch_dims_ = 0;
  edsl::InterpolationMode interpolation_mode_ = edsl::InterpolationMode::NONE;
};

inline edsl::Tensor hard_sigmoid(const edsl::Tensor& I, double slope) {
  auto args = edsl::make_tuple(I, slope);
  return details::op("hard_sigmoid", args).as_tensor();
}

inline edsl::Tensor image_resize(const edsl::Tensor& I, const std::vector<int>& factors,
                                 InterpolationMode interpolation, TensorLayout layout) {
  auto args = edsl::make_tuple(I, edsl::make_tuple(factors), static_cast<int>(interpolation), static_cast<int>(layout));
  return details::op("image_resize", args).as_tensor();
}

class lrn {
 public:
  explicit lrn(const edsl::Tensor& I, const std::vector<int64_t>& window_size)
      : I_(I), window_size_(window_size), axes_({-1}), alpha_(1.), beta_(1.), epsilon_(1.e-5) {}

  lrn& alpha(double alpha) {
    alpha_ = alpha;
    return *this;
  }

  lrn& beta(double beta) {
    beta_ = beta;
    return *this;
  }

  lrn& epsilon(double epsilon) {
    epsilon_ = epsilon;
    return *this;
  }

  lrn& window_size(const std::vector<int64_t>& window_size) {
    window_size_ = window_size;
    return *this;
  }

  lrn& axes(const std::vector<int64_t>& axes) {
    axes_ = axes;
    return *this;
  }

  operator edsl::Tensor() const {
    auto args = edsl::make_tuple(I_, edsl::make_tuple(window_size_), edsl::make_tuple(axes_), alpha_, beta_, epsilon_);
    return details::op("lrn", args).as_tensor();
  }

 private:
  edsl::Tensor I_;
  std::vector<int64_t> window_size_;
  std::vector<int64_t> axes_;
  double alpha_;
  double beta_;
  double epsilon_;
};

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

class mvn {
 public:
  explicit mvn(const plaidml::edsl::Tensor& I)
      : I_(I), axes_(edsl::None()), normalize_variance_(true), epsilon_(1e-9), across_channels_(true) {}

  mvn& axes(const std::vector<int64_t>& axes) {
    // negative axes interpreted as in numpy
    if (!across_channels_ || !layout_.empty()) {
      throw std::runtime_error("When using layout and across_channels, axes may not be specified");
    }
    axes_ = edsl::make_tuple(axes);
    return *this;
  }

  mvn& across_channels(bool flag) {
    if (!axes_.is_none()) {
      throw std::runtime_error("May not specify both axes and across_channels for MVN");
    }
    across_channels_ = flag;
    return *this;
  }

  mvn& normalize_variance(bool flag) {
    normalize_variance_ = flag;
    return *this;
  }

  mvn& epsilon(double value) {
    epsilon_ = value;
    return *this;
  }

  mvn& layout(const std::string& value) {
    if (!axes_.is_none()) {
      throw std::runtime_error("May not specify both axes and layout for MVN");
    }
    layout_ = value;
    return *this;
  }

  operator plaidml::edsl::Tensor() const {
    auto args = edsl::make_tuple(I_, axes_, normalize_variance_, epsilon_, across_channels_, layout_);
    return details::op("mvn", args).as_tensor();
  }

 private:
  plaidml::edsl::Tensor I_;
  edsl::Value axes_;
  bool normalize_variance_;
  double epsilon_;
  bool across_channels_;
  std::string layout_;
};

// NMS
// params
// @nms_style: There are currently 3 styles of nms. OpenVINO, Caffe and MxNet. OV style is the most common one and
//             the other two are fast versions of NMS which get the top k scores before nms. MxNet style only gets
//             the max score of each box.

class nms {
 public:
  explicit nms(edsl::Tensor Boxes, edsl::Tensor Scores, edsl::Tensor IOU_threshold, edsl::Tensor Score_threshold,
               int max_output_boxes_per_class)
      : Boxes_(Boxes),
        Scores_(Scores),
        IOU_threshold_(IOU_threshold),
        Score_threshold_(Score_threshold),
        max_output_boxes_per_class_(max_output_boxes_per_class),
        soft_nms_sigma_(0.0f),
        center_point_box_(false),
        sort_result_descending_(false),
        box_output_type_(DType::INT32),
        boxes_decode_mode_(BoxesDecodeMode::NMS),
        clip_before_nms_(false),
        clip_after_nms_(false),
        ssd_input_height_(0.0f),
        ssd_input_width_(0.0f),
        ssd_with_arm_loc_(false),
        nms_style_(NmsStyle::OV),
        share_location_(true),
        hard_suppression_(true) {}

  nms& soft_nms_sigma(float soft_nms_sigma) {
    soft_nms_sigma_ = soft_nms_sigma;
    return *this;
  }

  nms& center_point_box(bool center_point_box) {
    center_point_box_ = center_point_box;
    return *this;
  }

  nms& sort_result_descending(bool sort_result_descending) {
    sort_result_descending_ = sort_result_descending;
    return *this;
  }

  nms& box_output_type(DType box_output_type) {
    box_output_type_ = box_output_type;
    return *this;
  }

  nms& boxes_decode_mode(BoxesDecodeMode boxes_decode_mode) {
    boxes_decode_mode_ = boxes_decode_mode;
    return *this;
  }

  nms& clip_before_nms(bool clip_before_nms) {
    clip_before_nms_ = clip_before_nms;
    return *this;
  }

  nms& clip_after_nms(bool clip_after_nms) {
    clip_after_nms_ = clip_after_nms;
    return *this;
  }

  nms& ssd_input_height(int ssd_input_height) {
    ssd_input_height_ = ssd_input_height;
    return *this;
  }

  nms& ssd_input_width(int ssd_input_width) {
    ssd_input_width_ = ssd_input_width;
    return *this;
  }

  nms& ssd_variances(edsl::Tensor ssd_variances) {
    ssd_variances_ = ssd_variances;
    return *this;
  }

  nms& ssd_location(edsl::Tensor ssd_location) {
    ssd_location_ = ssd_location;
    return *this;
  }

  nms& ssd_arm_location(edsl::Tensor ssd_arm_location) {
    ssd_arm_location_ = ssd_arm_location;
    return *this;
  }

  nms& ssd_with_arm_loc(bool ssd_with_arm_loc) {
    ssd_with_arm_loc_ = ssd_with_arm_loc;
    return *this;
  }

  nms& nms_style(NmsStyle nms_style) {
    nms_style_ = nms_style;
    return *this;
  }

  nms& share_location(bool share_location) {
    share_location_ = share_location;
    return *this;
  }

  nms& hard_suppression(bool hard_suppression) {
    hard_suppression_ = hard_suppression;
    return *this;
  }

  std::vector<edsl::Tensor> build() {
    auto args = edsl::make_tuple(              //
        Boxes_,                                //
        Scores_,                               //
        IOU_threshold_,                        //
        Score_threshold_,                      //
        max_output_boxes_per_class_,           //
        soft_nms_sigma_,                       //
        center_point_box_,                     //
        sort_result_descending_,               //
        static_cast<int>(box_output_type_),    //
        static_cast<int>(boxes_decode_mode_),  //
        clip_before_nms_,                      //
        clip_after_nms_,                       //
        ssd_input_height_,                     //
        ssd_input_width_,                      //
        ssd_variances_,                        //
        ssd_location_,                         //
        ssd_with_arm_loc_,                     //
        ssd_arm_location_,                     //
        static_cast<int>(nms_style_),          //
        share_location_,                       //
        hard_suppression_);
    auto R = details::op("nms", args).as_tuple();
    auto B = R[0].as_tensor();
    auto S = R[1].as_tensor();
    auto V = R[2].as_tensor();
    return {B, S, V};
  }

 private:
  edsl::Tensor Boxes_;
  edsl::Tensor Scores_;
  edsl::Tensor ssd_location_;
  edsl::Tensor ssd_variances_;
  edsl::Tensor ssd_arm_location_;
  edsl::Tensor IOU_threshold_;
  edsl::Tensor Score_threshold_;
  int max_output_boxes_per_class_;
  float soft_nms_sigma_;
  bool center_point_box_;
  bool sort_result_descending_;
  DType box_output_type_;
  BoxesDecodeMode boxes_decode_mode_;
  bool clip_before_nms_;
  bool clip_after_nms_;
  int ssd_input_height_;
  int ssd_input_width_;
  bool ssd_with_arm_loc_;
  NmsStyle nms_style_;
  bool share_location_;
  bool hard_suppression_;
};

class topk {
 public:
  explicit topk(edsl::Tensor I, int k) : I_(I), k_(k) {}

  topk& axis(int axis) {
    axis_ = axis;
    return *this;
  }

  topk& sort_direction(edsl::SortDirection sort_direction) {
    sort_direction_ = sort_direction;
    return *this;
  }

  topk& sort_type(TopKSortType sort_type) {
    sort_type_ = sort_type;
    return *this;
  }

  topk& index_element_type(DType index_element_type) {
    index_element_type_ = index_element_type;
    return *this;
  }

  std::vector<edsl::Tensor> build() {
    auto args = edsl::make_tuple(           //
        I_,                                 //
        k_,                                 //
        axis_,                              //
        static_cast<int>(sort_direction_),  //
        static_cast<int>(sort_type_),       //
        static_cast<int>(index_element_type_));
    auto R = details::op("topk", args).as_tuple();
    auto values = R[0].as_tensor();
    auto indices = R[1].as_tensor();
    return {values, indices};
  }

 private:
  edsl::Tensor I_;
  int k_;
  int axis_ = -1;
  edsl::SortDirection sort_direction_ = edsl::SortDirection::DESC;
  TopKSortType sort_type_ = TopKSortType::VALUE;
  DType index_element_type_ = DType::INT32;
};

class l2norm {
 public:
  explicit l2norm(const edsl::Tensor& I, const std::vector<int64_t> axes)
      : I_(I), axes_(axes), epsilon_(0), eps_mode_(EpsMode::ADD) {}

  l2norm& epsilon(float epsilon) {
    epsilon_ = epsilon;
    return *this;
  }

  l2norm& eps_mode(EpsMode eps_mode) {
    eps_mode_ = eps_mode;
    return *this;
  }

  operator edsl::Tensor() const {
    auto args = edsl::make_tuple(I_, edsl::make_tuple(axes_), epsilon_, static_cast<int>(eps_mode_));
    return details::op("l2norm", args).as_tensor();
  }

 private:
  edsl::Tensor I_;
  std::vector<int64_t> axes_;
  float epsilon_;
  EpsMode eps_mode_;
};

inline edsl::Tensor pool(                    //
    const edsl::Tensor I,                    //
    PoolMode pool_mode,                      //
    const std::vector<int>& pool_size,       //
    const std::vector<int>& strides,         //
    AutoPadMode autopad_mode,                //
    const std::vector<int>& manual_padding,  //
    TensorLayout input_layout,               //
    bool include_padding_in_avg = false,     //
    bool use_ceil_for_output_shape = false,  //
    const std::string& name = ""             //
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
      use_ceil_for_output_shape,         //
      name);
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
    threshold_ = edsl::Value(threshold);
    return *this;
  }

  relu& name(const std::string& name) {
    name_ = name;
    return *this;
  }

  operator edsl::Tensor() const {
    auto args = edsl::make_tuple(I_, alpha_, max_value_, threshold_, name_);
    return details::op("relu", args).as_tensor();
  }

 private:
  edsl::Tensor I_;
  edsl::Tensor alpha_;
  edsl::Tensor max_value_;
  edsl::Value threshold_;
  std::string name_;
};

inline edsl::Tensor reorg_yolo(const edsl::Tensor& I, int stride, bool decrease, const std::string& layout = "NCHW") {
  auto args = edsl::make_tuple(I, stride, decrease, layout);
  return details::op("reorg_yolo", args).as_tensor();
}

class repeat {
 public:
  explicit repeat(const edsl::Tensor& I) : I_(I) {}

  repeat& count(int count) {
    count_ = edsl::Value(count);
    return *this;
  }

  repeat& count(const edsl::TensorDim& count) {
    count_ = edsl::Value(count);
    return *this;
  }

  repeat& axis(int axis) {
    axis_ = edsl::Value(axis);
    return *this;
  }

  operator edsl::Tensor() const {
    auto args = edsl::make_tuple(I_, count_, axis_);
    return details::op("repeat", args).as_tensor();
  }

 private:
  edsl::Tensor I_;
  edsl::Value count_ = edsl::Value(1);
  edsl::Value axis_ = edsl::Value(0);
};

inline edsl::Tensor reshape(const edsl::Tensor& I, const edsl::Value& dims) {
  auto args = edsl::make_tuple(I, dims);
  return details::op("reshape", args).as_tensor();
}

inline edsl::Tensor sigmoid(const edsl::Tensor& I) {
  auto args = edsl::make_tuple(I);
  return details::op("sigmoid", args).as_tensor();
}

class slice {
 public:
  explicit slice(const edsl::Tensor& I) : I_(I) {}

  slice& add_dims(const std::vector<int>& dims) {
    for (auto dim : dims) {
      dims_.emplace_back(dim);
    }
    return *this;
  }

  slice& add_dim(int dim) {
    dims_.emplace_back(dim);
    return *this;
  }

  slice& add_dim(int start, int stop, int step = 1) {
    dims_.emplace_back(edsl::make_tuple(start, stop, step));
    return *this;
  }

  slice& add_dim(edsl::Value start, edsl::Value stop, edsl::Value step = edsl::Value(1)) {
    dims_.emplace_back(edsl::make_tuple(start, stop, step));
    return *this;
  }

  operator edsl::Tensor() const {
    auto args = edsl::make_tuple(I_, dims_);
    return details::op("slice", args).as_tensor();
  }

 private:
  edsl::Tensor I_;
  std::vector<edsl::Value> dims_;
};

inline edsl::Tensor softmax(const edsl::Tensor& I, int axis, const std::string& name = "") {
  auto args = edsl::make_tuple(I, axis, name);
  return details::op("softmax", args).as_tensor();
}

inline edsl::Tensor sort(const edsl::Tensor& I, int axis, edsl::SortDirection direction = edsl::SortDirection::ASC) {
  auto args = edsl::make_tuple(I, axis, static_cast<int>(direction));
  return details::op("sort", args).as_tensor();
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

inline edsl::Tensor unsqueeze(const edsl::Tensor& I, const std::vector<int64_t>& axes) {
  auto args = edsl::make_tuple(I, edsl::make_tuple(axes));
  return details::op("unsqueeze", args).as_tensor();
}

inline edsl::Tensor variance(const edsl::Tensor& I, const edsl::Value& axes = edsl::None(), bool keepdims = false) {
  auto args = edsl::make_tuple(I, axes, keepdims);
  return details::op("variance", args).as_tensor();
}

}  // namespace op
}  // namespace plaidml
