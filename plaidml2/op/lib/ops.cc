// Copyright 2019 Intel Corporation.

#include "plaidml2/op/lib/ops.h"

#include <boost/format.hpp>

#include "base/util/logging.h"
#include "plaidml2/op/op.h"

using namespace plaidml::edsl;  // NOLINT

namespace plaidml {
namespace op {
namespace lib {

struct AggregationAxes {
  std::vector<TensorIndex> src_idxs;
  std::vector<TensorIndex> dst_idxs;
  std::vector<TensorIndex> reduce_idxs;
  std::vector<TensorDim> src_dims;
  std::vector<TensorDim> dst_dims;
  std::vector<TensorDim> reduce_dims;
  std::set<size_t> axes;

  AggregationAxes(size_t ndims, const Value& in_axes, bool keepdims) : src_idxs(ndims), src_dims(ndims) {
    IVLOG(5, "Received agg axes request with\n\tndims = " << ndims << "\n\tin_axes = " << in_axes
                                                          << "\n\tkeepdims = " << keepdims);
    if (in_axes.is_none()) {
      for (size_t i = 0; i < ndims; i++) {
        axes.insert(i);
      }
    } else if (in_axes.is_tuple()) {
      for (const auto& axis : in_axes.as_tuple()) {
        auto int_axis = axis.as_int();
        if (int_axis < 0) {
          int_axis = ndims + int_axis;
        }
        if (int_axis < 0 || ndims < static_cast<size_t>(int_axis)) {
          throw std::out_of_range(str(boost::format("axis out of range: %1%") % int_axis));
        }
        axes.insert(int_axis);
      }
    } else if (in_axes.is_int()) {
      auto axis = in_axes.as_int();
      if (axis < 0) {
        axis = ndims + axis;
      }
      axes = {static_cast<size_t>(axis)};
    } else {
      throw std::runtime_error("Invalid Value type for AggregationAxes: in_axes");
    }
    for (auto axis : axes) {
      reduce_idxs.push_back(src_idxs[axis]);
      reduce_dims.push_back(src_dims[axis]);
    }
    if (keepdims) {
      dst_idxs = src_idxs;
      dst_dims = src_dims;
      for (auto axis : axes) {
        dst_idxs[axis] = TensorIndex();
        dst_dims[axis] = TensorDim{1};
      }
    } else {
      for (size_t i = 0; i < ndims; i++) {
        if (!axes.count(i)) {
          dst_idxs.push_back(src_idxs[i]);
          dst_dims.push_back(src_dims[i]);
        }
      }
    }
  }
};

enum class AutogroupMode {
  UNGROUPED,  // Group size explicitly 1
  EXPLICIT,   // Group size explicitly specified, > 1
  AUTO,       // Group size determined from shapes of I and F
  DEPTHWISE   // for channelized convolutions (i.e. where G = CI)
};

enum class AutopadMode : char {
  NONE = '-',
  NOTSET = NONE,
  EXPLICIT = NONE,
  SAME_LOWER = 'L',
  SAME_UPPER = 'U',
  VALID = 'V'
};

enum class ConvDerivMode {
  NONE,   // Forward Pass
  DATA,   // Computing derivative of input data (or equivalently a transposed conv)
  FILTER  // Computing derivative of filters
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
  IN_K       // Group included in the output channels dimensiono
};

enum class PoolMode : char { AVG = 'A', MAX = '>', MIN = '<', SUM = '+' };

enum class TensorLayout { NXC, NCX, KCX, XCK, GKCX, XGCK };

namespace {
// TODO: I haven't decided whether to make these helper functions visible to the outside world

AutogroupMode autogroup_mode_from_str(const std::string& s) {
  if (s == "ungrouped") {
    return AutogroupMode::UNGROUPED;
  }
  if (s == "explicit") {
    return AutogroupMode::EXPLICIT;
  }
  if (s == "auto") {
    return AutogroupMode::AUTO;
  }
  if (s == "max") {
    return AutogroupMode::DEPTHWISE;
  }
  throw std::runtime_error(str(boost::format("Unable to parse string '%1%' as an autogroup mode") % s));
}

AutopadMode autopad_mode_from_str(const std::string& s) {
  if (s == "none") {
    return AutopadMode::NONE;
  }
  if (s == "notset") {
    return AutopadMode::NOTSET;
  }
  if (s == "explicit") {
    return AutopadMode::EXPLICIT;
  }
  if (s == "same_lower") {
    return AutopadMode::SAME_LOWER;
  }
  if (s == "same_upper") {
    return AutopadMode::SAME_UPPER;
  }
  if (s == "valid") {
    return AutopadMode::VALID;
  }
  throw std::runtime_error(str(boost::format("Unable to parse string '%1%' as an autopadding mode") % s));
}

std::string to_string(AutopadMode m) {
  switch (m) {
    case AutopadMode::NONE:
      return "none";
    case AutopadMode::SAME_LOWER:
      return "same_lower";
    case AutopadMode::SAME_UPPER:
      return "same_upper";
    case AutopadMode::VALID:
      return "valid";
  }
  throw std::runtime_error("Unable to convert autopadding mode to string due to unrecognized mode");
}

ConvDerivMode conv_deriv_mode_from_str(const std::string& s) {
  if (s == "none") {
    return ConvDerivMode::NONE;
  }
  if (s == "data") {
    return ConvDerivMode::DATA;
  }
  if (s == "filter") {
    return ConvDerivMode::FILTER;
  }
  throw std::runtime_error(str(boost::format("Unable to parse string '%1%' as a convolution derivative mode") % s));
}

GroupLayout group_layout_from_str(const std::string& s) {
  if (s == "none") {
    return GroupLayout::NONE;
  }
  if (s == "in_C") {
    return GroupLayout::IN_C;
  }
  if (s == "in_K") {
    return GroupLayout::IN_K;
  }
  if (s == "separate") {
    return GroupLayout::SEPARATE;
  }
  throw std::runtime_error(str(boost::format("Unable to parse string '%1%' as a group layout") % s));
}

std::string to_string(GroupLayout l) {
  switch (l) {
    case GroupLayout::IN_C:
      return "in_C";
    case GroupLayout::IN_K:
      return "in_K";
    case GroupLayout::NONE:
      return "none";
    case GroupLayout::SEPARATE:
      return "separate";
  }
  throw std::runtime_error("Unable to convert group layout to string due to unrecognized layout");
}

PoolMode pool_mode_from_str(const std::string& s) {
  if (s == "avg" || s == "average") {
    return PoolMode::AVG;
  }
  if (s == "max") {
    return PoolMode::MAX;
  }
  if (s == "min") {
    return PoolMode::MIN;
  }
  if (s == "sum") {
    return PoolMode::SUM;
  }
  throw std::runtime_error(str(boost::format("Unable to parse string '%1%' as a pooling mode") % s));
}

// TODO: Enable when needed
// std::string to_string(PoolMode m) {
//   switch (m) {
//    case PoolMode::AVG:
//     return "avg";
//    case PoolMode::MAX:
//     return "max";
//    case PoolMode::MIN:
//     return "min";
//    case PoolMode::SUM:
//     return "sum";
//   }
//   throw std::runtime_error("Unable to convert pooling mode to string due to unrecognized mode");
// }

TensorLayout tensor_layout_from_str(const std::string& s) {
  if (s == "nxc" || s == "nwc" || s == "nhwc" || s == "ndhwc") {
    return TensorLayout::NXC;
  }
  if (s == "ncx" || s == "ncw" || s == "nchw" || s == "ncdhw") {
    return TensorLayout::NCX;
  }
  if (s == "kcx" || s == "kcw" || s == "kchw" || s == "kcdhw") {
    return TensorLayout::KCX;
  }
  if (s == "xck" || s == "wck" || s == "hwck" || s == "dhwck") {
    return TensorLayout::XCK;
  }
  if (s == "gkcx" || s == "gkcw" || s == "gkchw" || s == "gkcdhw") {
    return TensorLayout::GKCX;
  }
  if (s == "xgck" || s == "wgck" || s == "hwgck" || s == "dhwgck") {
    return TensorLayout::XGCK;
  }
  throw std::runtime_error(str(boost::format("Unable to parse string '%1%' as a tensor layout") % s));
}

size_t nonspatial_dims(TensorLayout layout) {
  switch (layout) {
    case TensorLayout::NXC:
    case TensorLayout::NCX:
    case TensorLayout::KCX:
    case TensorLayout::XCK:
      return 2;
    case TensorLayout::GKCX:
    case TensorLayout::XGCK:
      return 3;
    default:
      throw std::runtime_error("Unrecognized layout when attempting to count non-spatial dimensions");
  }
}

// TODO: Enable when needed
// std::string to_string(TensorLayout m) {
//   switch (m) {
//    case TensorLayout::NXC:
//     return "NXC";
//    case TensorLayout::NCX:
//     return "NCX";
//    case TensorLayout::KCX:
//     return "KCX";
//    case TensorLayout::XCK:
//     return "XCK";
//    case TensorLayout::GKCX:
//     return "GKCX";
//    case TensorLayout::XGCK:
//     return "XGCK";
//   }
//   throw std::runtime_error("Unable to convert tensor layout to string due to unrecognized layout");
// }

bool is_input_layout(TensorLayout layout) {  //
  return (layout == TensorLayout::NCX || layout == TensorLayout::NXC);
}

bool is_filter_layout(TensorLayout layout) {
  return (layout == TensorLayout::KCX || layout == TensorLayout::XCK || layout == TensorLayout::GKCX ||
          layout == TensorLayout::XGCK);
}

bool is_filter_layout_with_separate_groups(TensorLayout layout) {
  return (layout == TensorLayout::GKCX || layout == TensorLayout::XGCK);
}

void normalize_grouping_strategy(int64_t* groups, AutogroupMode* autogroup_mode, GroupLayout* group_layout) {
  // This normalization enforces:
  //  * If group_layout is NONE:
  //      - autogroup_mode is UNGROUPED
  //        (AUTO is converted to UNGROUPED, as is EXPLICIT if groups == 1)
  //        (DEPTHWISE throws, as does EXPLICIT if groups != 1)
  //      - groups is 1
  //        (If non-1, it is converted after autogroup_mode conversion succeeds)
  //  * If autogroup_mode is UNGROUPED:
  //      - groups is 1
  //        (If non-1, it throws)
  //      - group_layout allowed to vary
  //  * If autogroup_mode is EXPLICIT:
  //      - groups is > 1
  //        (If < 1, throw; if == 1, autogroup_mode is converted to UNGROUPED)
  //      - group_layout is allowed to vary, but may not be NONE
  //        (If group_layout is NONE and groups != 1, this throws (see above))
  //        (If group_layout is NONE and groupd == 1, autogroup_mode is converted to UNGROUPED (see above))
  //  * If autogroup_mode is AUTO:
  //      - groups is to be ignored
  //      - group_layout is SEPARATE or IN_K
  //        (throws if group_layout is IN_C)
  //        (if group_layout is NONE, autogroup_mode is converted to UNGROUPED (see above))
  //  * If autogroup_mode is DEPTHWISE:
  //      - groups is to be ignored
  //      - group_layout is not NONE
  switch (*autogroup_mode) {
    case AutogroupMode::UNGROUPED:
      if (*groups != 1) {
        throw std::runtime_error("Convolution AutogroupMode::UNGROUPED requires groups == 1");
      }
      break;
    case AutogroupMode::AUTO:
      if (*group_layout == GroupLayout::NONE) {
        *groups = 1;
        *autogroup_mode = AutogroupMode::UNGROUPED;
      }
      if (*group_layout == GroupLayout::IN_C) {
        // TODO: This and related cases may depend on the deriv_mode; take that into account
        throw std::runtime_error("Cannot automatically detect group size of convolution with IN_C GroupLayout");
      }
      break;
    case AutogroupMode::EXPLICIT:
      if (*groups < 1) {
        throw std::runtime_error("Requested grouped convolution with fewer than 1 groups");
      }
      if (*groups == 1) {
        *autogroup_mode = AutogroupMode::UNGROUPED;
      }
      if (*group_layout == GroupLayout::NONE && *groups != 1) {
        throw std::runtime_error("GroupLayout not specified for grouped convolution");
      }
      break;
    case AutogroupMode::DEPTHWISE:
      if (*group_layout == GroupLayout::NONE) {
        throw std::runtime_error("Convolution GroupLayout must be specified to use DEPTHWISE AutogroupMode");
      }
      break;
    default:
      throw std::runtime_error("Unrecognized AutogroupMode");
  }
}

size_t normalize_axis(int64_t axis, size_t ndims, std::string op_name = "") {
  bool negative_axis_given = false;
  if (axis < 0) {
    axis += ndims;
    negative_axis_given = true;
  }
  if (axis < 0 || ndims <= static_cast<size_t>(axis)) {
    if (negative_axis_given) {
      axis -= ndims;
    }
    auto op_name_str = op_name.empty() ? "" : str(boost::format("for %1% op ") % op_name);
    throw std::runtime_error(
        str(boost::format("Axis out of range %1%(axis %2% requested for tensors with %3% dimensions)") % op_name_str %
            axis % ndims));
  }
  return axis;
}

std::pair<TensorDim, TensorDim> compute_padding_and_output_size(const TensorDim& input_size,
                                                                const TensorDim& filter_size, int64_t stride,
                                                                AutopadMode autopad_mode, int64_t pad_lo,
                                                                int64_t pad_hi, int64_t dilation, int64_t data_dilation,
                                                                bool use_ceil_for_output_shape) {
  // Effective input and filter sizes are the sizes after dilations are accounted for. So a 4x3 filter dilated by (3, 2)
  // has an effective filter size of 11 and 5 for its 2 spatial dims

  auto I_eff = (data_dilation * (input_size - 1)) + 1;  // Effective Input Size
  auto F_eff = (dilation * (filter_size - 1)) + 1;      // Effective Filter Size
  int64_t ceil_term =
      use_ceil_for_output_shape ? stride - 1 : 0;  // TODO: Will need to confirm that this is the intended behavior
  if (autopad_mode == AutopadMode::NONE) {
    TensorDim pad_before(pad_lo);
    TensorDim output_size((I_eff + pad_lo + pad_hi - F_eff + stride + ceil_term) / stride);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  }
  if (autopad_mode == AutopadMode::VALID) {
    TensorDim pad_before(0);
    TensorDim output_size((I_eff - F_eff + stride + ceil_term) / stride);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  }
  if (autopad_mode == AutopadMode::SAME_LOWER || autopad_mode == AutopadMode::SAME_UPPER) {
    TensorDim output_size((I_eff + stride - 1 + ceil_term) / stride);
    int64_t lower_term = (autopad_mode == AutopadMode::SAME_LOWER) ? 1 : 0;
    // TensorDim pad_before((max(0, (output_size - 1) * stride + F_eff - I_eff) + upper_term) / 2);
    // TODO: Switch to above once max(TensorDim, TensorDim) is working
    TensorDim pad_before(((output_size - 1) * stride + F_eff - I_eff + lower_term) / 2);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  }
  throw std::runtime_error(str(boost::format("Unexpected autopadding mode: %1%") % to_string(autopad_mode)));
}

std::vector<int64_t>* extend_manual_padding(std::vector<int64_t>* pads, size_t rank) {
  // TODO: Perhaps we should throw for sizes != 0, rank, 2*rank?
  if (pads->size() > 2 * rank) {
    throw std::runtime_error(str(
        boost::format(
            "Inconsistent spatial rank: operation with %1% spatial dimensions had %2% manual padding values given") %
        rank % pads->size()));
  }
  while (pads->size() < rank) {
    pads->push_back(0);
  }
  while (pads->size() < 2 * rank) {
    // Where pad_hi isn't set, copy the pad_lo value
    pads->push_back(pads->at(pads->size() - rank));
  }
  return pads;
}

}  // namespace

Value abs(const Value& value) {
  IVLOG(1, "abs");
  auto args = value.as_tuple();
  if (args.size() != 1) {
    throw std::runtime_error("abs expects 1 argument");
  }
  auto I = args[0].as_tensor();
  auto O = select(I < 0., -I, I);
  return Value{O};
}

Value argmax(const Value& value) {
  IVLOG(1, "argmax");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("argmax expects 2 arguments");
  }
  auto I = args[0].as_tensor();
  auto I_shape = I.shape();
  auto axes = args[1];
  AggregationAxes agg(I_shape.ndims(), axes, false);
  I.bind_dims(agg.src_dims);
  auto M = TensorOutput(agg.dst_dims);
  M(agg.dst_idxs) >= I(agg.src_idxs);
  Tensor One(1);
  auto T = TensorOutput(agg.reduce_dims);
  T(agg.reduce_idxs) = One();
  auto IX = index(T, 0);
  auto AM = TensorOutput(agg.dst_dims);
  AM(agg.dst_idxs) >= cond(I(agg.src_idxs), M(agg.dst_idxs), IX(agg.reduce_idxs));
  auto O = as_uint(AM, 32);
  return Value{O};
}

Value concatenate(const Value& value) {
  // TODO: Make errors nicer (e.g. when bind_dims fails)
  IVLOG(1, "concatenate")

  // Read Arguments
  auto args = value.as_tuple();
  auto tensor_vals = args[0].as_tuple();
  auto raw_axis = args[1].as_int();

  // Initialize useful values
  std::vector<Tensor> tensors;
  for (auto tensor_val : tensor_vals) {
    tensors.push_back(tensor_val.as_tensor());
  }
  if (tensors.empty()) {
    throw std::runtime_error("The concatenate op requires at least one input tensor");
  }
  auto ndims = tensors[0].shape().ndims();
  auto axis = normalize_axis(raw_axis, ndims, "concatenate");
  std::vector<TensorDim> dims(ndims);
  std::vector<TensorIndex> I_idxs;  // These will be named
  std::vector<TensorIndex> O_idxs;
  std::vector<TensorDim> axis_dims(tensors.size());
  std::vector<TensorDim> axis_dim_subtotals;
  std::vector<Tensor> results;
  TensorIndex axis_idx("a");
  axis_dim_subtotals.emplace_back(0);
  for (size_t i = 0; i < ndims; ++i) {
    if (i == axis) {
      I_idxs.push_back(axis_idx);
    } else {
      I_idxs.emplace_back(str(boost::format("n%1%") % i));
    }
  }

  // Bind the various input dimensions
  for (size_t i = 0; i < tensors.size(); ++i) {
    dims[axis] = axis_dims[i];
    tensors[i].bind_dims(dims);
    axis_dim_subtotals.push_back(axis_dims[i] + axis_dim_subtotals.back());
  }
  // Set dims to the final output dims
  dims[axis] = axis_dim_subtotals[tensors.size()];  // There are tensors.size() + 1 TensorDims in the RHS vector
  O_idxs = I_idxs;                                  // The axis index will be overwritten during the loop

  // Compute each intermediate output
  for (size_t i = 0; i < tensors.size(); ++i) {
    results.emplace_back(TensorOutput(dims));
    O_idxs[axis] = axis_idx + axis_dim_subtotals[i];
    results[i](O_idxs) = tensors[i](I_idxs);
    if (i > 0) {
      results[i].use_default(results[i - 1]);
    }
  }

  return Value{results[tensors.size() - 1]};
}

Value convolution(const Value& value) {
  IVLOG(1, "convolution");
  // Parameters:
  //  0. Input Tensor
  //  1. Filter Tensor
  //  2. Strides
  //  3. Dilations
  //  4. Data Dilations
  //  5. Kernel Shape
  //  6. Groups
  //  7. Autopad Mode
  //  8. Manual Padding
  //  9. Input Tensor Layout
  // 10. Filter Tensor Layout
  // 11. Grouping Layout
  // 12. Winograd allowed
  // 13. Name
  // 14. Autogrouping (? Unclear if we really need this)
  // 15. Deriv Mode (DATA is equivalent to transposed conv)
  // 16. Result Shape (a.k.a. output shape, used for transposed/derivative convs)

  // Read Arguments
  auto args = value.as_tuple();
  if (args.size() != 17) {
    throw std::runtime_error("Convolution op expects 17 arguments");
  }
  auto I_or_O = args[0].as_tensor();  // O if deriv_mode is DATA, else I
  auto F_or_O = args[1].as_tensor();  // O if deriv_mode is FILTER, else F
  auto strides = args[2].as_int_tuple();
  auto dilations = args[3].as_int_tuple();
  auto data_dilations = args[4].as_int_tuple();
  // TODO: Perhaps could upgrade use of filter_shape?
  auto filter_shape = args[5].as_int_tuple();  // This is the shape of the _spatial_ filter dims _only_
  auto groups = args[6].as_int();              // will be 1 for non-grouped convolutions
  auto autopad_mode = autopad_mode_from_str(args[7].as_str());
  auto manual_padding = args[8].as_int_tuple();
  auto input_layout = tensor_layout_from_str(args[9].as_str());
  auto filter_layout = tensor_layout_from_str(args[10].as_str());
  auto group_layout = group_layout_from_str(args[11].as_str());
  // auto winograd_allowed = args[12].as_bool();  // TODO: Implement Winograd
  auto name = args[13].as_str();
  auto autogroup_mode = autogroup_mode_from_str(args[14].as_str());
  auto deriv_mode = conv_deriv_mode_from_str(args[15].as_str());
  auto result_shape = args[16].as_int_tuple();

  Tensor I;       // Inputs (i.e. Data) tensor
  Tensor F;       // Filters (i.e. Weights i.e. Kernel) tensor
  Tensor O;       // Output (i.e. of a forward pass) tensor
  Tensor Result;  // The tensor that is actually returned, depends on deriv_mode

  // Connect the inputs to the right names
  switch (deriv_mode) {
    case ConvDerivMode::NONE:
      I = I_or_O;
      F = F_or_O;
      break;
    case ConvDerivMode::DATA:
      O = I_or_O;
      F = F_or_O;
      break;
    case ConvDerivMode::FILTER:
      I = I_or_O;
      O = F_or_O;
      break;
  }

  // Initialize useful values
  auto spatial_rank = strides.size();

  // Verify inputs are consistent
  if (manual_padding.size() && autopad_mode != AutopadMode::NONE) {
    throw std::runtime_error("Autopadding and manual padding both requested for single conv operation");
  }
  if (dilations.size() != spatial_rank) {
    throw std::runtime_error(
        str(boost::format("Inconsistent spatial rank in conv op (received %1%D strides and %2%D dilations)") %
            strides.size() % dilations.size()));
  }
  if (data_dilations.size() != spatial_rank) {
    throw std::runtime_error(
        str(boost::format("Inconsistent spatial rank in conv op (received %1%D strides and %2%D data_dilations)") %
            strides.size() % data_dilations.size()));
  }
  if (!is_input_layout(input_layout)) {
    throw std::runtime_error("Input tensor layout requested in conv op does not apply to convolution input tensors");
  }
  if (!is_filter_layout(filter_layout)) {
    throw std::runtime_error("Filter tensor layout requested in conv op does not apply to convolution filter tensors");
  }
  if (deriv_mode != ConvDerivMode::DATA && I.shape().ndims() - spatial_rank != nonspatial_dims(input_layout)) {
    // If we ever extend possible layouts so that I and O may have different layouts, we will
    // need to do this check in different ways depending on whether deriv_mode is DATA or not
    throw std::runtime_error(
        str(boost::format("Inconsistent spatial rank in conv op (received %1% spatial dimensions based on strides but "
                          "input tensor has %2% dimensions, and thus %3% spatial dims). (This error can also occur if "
                          "the layout of I is incorrectly specified or interpreted.)") %
            spatial_rank % I.shape().ndims() % (I.shape().ndims() - nonspatial_dims(input_layout))));
  }
  if (deriv_mode != ConvDerivMode::FILTER && F.shape().ndims() - spatial_rank != nonspatial_dims(filter_layout)) {
    throw std::runtime_error(str(
        boost::format("Inconsistent spatial rank in conv op (received %1% spatial dimensions based on strides "
                      "but filter tensor has %2% dimensions, and thus %3% spatial dims). (This error can also occur "
                      "if the layout of F is incorrectly specified or interpreted.)") %
        spatial_rank % F.shape().ndims() % (F.shape().ndims() - nonspatial_dims(filter_layout))));
  }
  if (filter_shape.size() && (filter_shape.size() != spatial_rank)) {
    throw std::runtime_error(
        str(boost::format("Filter shape manually specified with inconsistent rank (received %1% spatial dimensions "
                          "based on strides but filter_shape has %2% dimensions)") %
            spatial_rank % filter_shape.size()));
  }
  if (is_filter_layout_with_separate_groups(filter_layout) && group_layout != GroupLayout::SEPARATE) {
    throw std::runtime_error("Filter_layout specifies separate groups but group_layout isn't SEPARATE");
  }
  if (!is_filter_layout_with_separate_groups(filter_layout) && group_layout == GroupLayout::SEPARATE) {
    throw std::runtime_error("Filter_layout lacks separate groups but group_layout is SEPARATE");
  }
  if (result_shape.size() == 0) {
    if (deriv_mode != ConvDerivMode::NONE) {
      throw std::runtime_error("Transposed/gradient convolutions require specifying the result_shape");
    }
  } else {
    if (result_shape.size() != spatial_rank) {
      throw std::runtime_error(
          str(boost::format("Inconsistent spatial rank in conv op (received %1% spatial dimensions based on strides "
                            "but result shape has %2% spatial dims).") %
              spatial_rank % result_shape.size()));
    }
  }

  // Fill in defaults & normalize inputs
  extend_manual_padding(&manual_padding, spatial_rank);
  if (name.empty()) {
    name = "conv";
  }
  normalize_grouping_strategy(&groups, &autogroup_mode, &group_layout);

  // Prepare dimension and index variables
  TensorDim N, CI, CO, G;
  TensorDim F_CI, F_CO;  // The channel dimensions as used by the filters, adjusted for group layout
  TensorIndex n("n");
  TensorIndex ci("ci");
  TensorIndex co("co");
  TensorIndex g("g");
  std::vector<TensorDim> X(spatial_rank);  // The spatial dimensions of I
  std::vector<TensorIndex> x;              // The spatial indexes of I
  for (size_t i = 0; i < spatial_rank; ++i) {
    x.emplace_back(TensorIndex(str(boost::format("x%1%") % i)));
  }
  std::vector<TensorDim> Y(spatial_rank);  // The spatial dimensions of O; nearly unused
  std::vector<TensorDim> K(spatial_rank);  // The spatial dimensions of F
  std::vector<TensorIndex> k;              // The spatial indexs of F
  for (size_t i = 0; i < spatial_rank; ++i) {
    k.emplace_back(TensorIndex(str(boost::format("k%1%") % i)));
  }
  std::vector<TensorDim> I_dims;
  std::vector<TensorIndex> I_idxs;
  std::vector<TensorDim> F_dims;
  std::vector<TensorDim>
      F_explicit_dims;  // this ensures that the inferred filter shape matches filter_shape if the latter is passed in
  std::vector<TensorIndex> F_idxs;
  std::vector<TensorDim> O_dims;
  std::vector<TensorIndex> O_idxs;
  TensorDim G_explicit(groups);  // G may be explicit or automatically set, based on autogroup_mode
  switch (autogroup_mode) {
    case AutogroupMode::EXPLICIT:
    case AutogroupMode::UNGROUPED:
      G = G_explicit;
      break;
    case AutogroupMode::DEPTHWISE:
      G = CI;
      if (group_layout == GroupLayout::IN_K || group_layout == GroupLayout::SEPARATE) {
        F_CI = TensorDim(1);
      } else if (group_layout == GroupLayout::IN_C) {
        // Everything can be inferred, do nothing  // nolint(whitespace/empty_if_body)
      } else {
        throw std::runtime_error(
            str(boost::format("Unsupported group layout '%1%' used with autogroup mode DEPTHWISE") %
                to_string(group_layout)));
      }
      break;
    case AutogroupMode::AUTO:
      if (group_layout == GroupLayout::SEPARATE || group_layout == GroupLayout::IN_K) {
        // just let G be inferred; i.e. do nothing  // nolint(whitespace/empty_if_body)
      } else {
        throw std::runtime_error(str(boost::format("Unsupported group layout '%1%' used with autogroup mode AUTO") %
                                     to_string(group_layout)));
      }
      break;
    default:
      throw std::runtime_error("Unrecognized AutogroupMode");
  }

  // Set up dimensions of the inputs first so they can be bound
  // Group layout affects the size of filter dimensions; we pass through the dims that don't need to be adjusted here,
  // and we will calculate those dimensions that will be adjusted later (after some more dims are bound).
  // TODO: This needs more thorough test converage
  switch (group_layout) {
    case GroupLayout::NONE:
      F_CO = CO;
      F_CI = CI;
      break;
    case GroupLayout::IN_C:
      // Later: F_CO = CO / G;
      F_CI = CI;
      break;
    case GroupLayout::IN_K:
      F_CO = CO;
      // Later: F_CI = CI / G;
      break;
    case GroupLayout::SEPARATE:
      // Later: F_CO = CO / G;
      // Later: F_CI = CI / G;
      break;
  }

  // The input data dims
  if (deriv_mode != ConvDerivMode::DATA) {
    switch (input_layout) {
      case TensorLayout::NCX:
        I_dims.push_back(N);
        I_dims.push_back(CI);
        for (size_t i = 0; i < spatial_rank; ++i) {
          I_dims.push_back(X[i]);
        }
        break;
      case TensorLayout::NXC:
        I_dims.push_back(N);
        for (size_t i = 0; i < spatial_rank; ++i) {
          I_dims.push_back(X[i]);
        }
        I_dims.push_back(CI);
        break;
      default:
        throw std::runtime_error("Invalid input_layout");
    }
    I.bind_dims(I_dims);
  }

  // The filter dims
  if (deriv_mode != ConvDerivMode::FILTER) {
    switch (filter_layout) {
      case TensorLayout::GKCX:
        F_dims.push_back(G);
        F_explicit_dims.push_back(G);
        // Fall through deliberately
      case TensorLayout::KCX:
        F_dims.push_back(F_CO);
        F_explicit_dims.push_back(F_CO);
        F_dims.push_back(F_CI);
        F_explicit_dims.push_back(F_CI);
        for (size_t i = 0; i < spatial_rank; ++i) {
          F_dims.push_back(K[i]);
          if (filter_shape.size()) {
            F_explicit_dims.push_back(TensorDim(filter_shape[i]));
          }
        }
        break;
      case TensorLayout::XCK:
      case TensorLayout::XGCK:
        for (size_t i = 0; i < spatial_rank; ++i) {
          F_dims.push_back(K[i]);
          if (filter_shape.size()) {
            F_explicit_dims.push_back(TensorDim(filter_shape[i]));
          }
        }
        if (filter_layout == TensorLayout::XGCK) {
          F_dims.push_back(G);
          F_explicit_dims.push_back(G);
        }
        F_dims.push_back(F_CI);
        F_explicit_dims.push_back(F_CI);
        F_dims.push_back(F_CO);
        F_explicit_dims.push_back(F_CO);
        break;
      default:
        throw std::runtime_error("Invalid filter_layout");
    }
    F.bind_dims(F_dims);
    if (filter_shape.size()) {
      F.bind_dims(F_explicit_dims);
    }
  }

  // The output data dims
  if (deriv_mode != ConvDerivMode::NONE) {
    // This assumes we infer the output layout from the input layout. So if we change that, the output data dims section
    // will need to be adapted.
    switch (input_layout) {
      case TensorLayout::NCX:
        O_dims.push_back(N);
        O_dims.push_back(CO);
        for (size_t i = 0; i < spatial_rank; ++i) {
          O_dims.push_back(Y[i]);
        }
        break;
      case TensorLayout::NXC:
        O_dims.push_back(N);
        for (size_t i = 0; i < spatial_rank; ++i) {
          O_dims.push_back(Y[i]);
        }
        O_dims.push_back(CO);
        break;
      default:
        throw std::runtime_error("Invalid input_layout");
    }
    O.bind_dims(O_dims);
  }

  // Compute the adjustments to the filter channel dimensions that come from group size
  switch (group_layout) {
    case GroupLayout::NONE:
      break;
    case GroupLayout::IN_C:
      CO = F_CO * G;
      break;
    case GroupLayout::IN_K:
      CI = F_CI * G;
      break;
    case GroupLayout::SEPARATE:
      CO = F_CO * G;
      CI = F_CI * G;
      break;
  }

  // Determine the padding and the shape of the result tensor
  std::vector<TensorDim> pad_before;
  std::vector<TensorDim> O_spatial_dims;
  for (size_t i = 0; i < spatial_rank; ++i) {
    TensorDim local_pad_before;
    TensorDim local_output_size;
    TensorDim local_input_size = (deriv_mode == ConvDerivMode::DATA) ? TensorDim(result_shape[i]) : X[i];
    TensorDim local_filter_size = (deriv_mode == ConvDerivMode::FILTER) ? TensorDim(result_shape[i]) : K[i];
    std::tie(local_pad_before, local_output_size) = compute_padding_and_output_size(
        local_input_size, local_filter_size, strides[i], autopad_mode, manual_padding[i],
        manual_padding[i + spatial_rank], dilations[i], data_dilations[i], false);
    pad_before.emplace_back(local_pad_before);
    O_spatial_dims.emplace_back(local_output_size);
  }

  // Now set up the dimensions of the result to be returned
  switch (deriv_mode) {
    case ConvDerivMode::NONE:
      // This assumes we infer the output layout from the input layout. So if we change that, the below switch will need
      // to be adapted.
      switch (input_layout) {
        case TensorLayout::NCX:
          O_dims.push_back(N);
          O_dims.push_back(CO);
          for (size_t i = 0; i < spatial_rank; ++i) {
            O_dims.push_back(O_spatial_dims[i]);
          }
          break;
        case TensorLayout::NXC:
          O_dims.push_back(N);
          for (size_t i = 0; i < spatial_rank; ++i) {
            O_dims.push_back(O_spatial_dims[i]);
          }
          O_dims.push_back(CO);
          break;
        default:
          throw std::runtime_error("Invalid input_layout");
      }
      O = Tensor{name, O_dims};
      // O = NamedTensorOutput(name, O_dims);  // TODO: Re-enable when ready
      break;
    case ConvDerivMode::DATA:
      switch (input_layout) {
        case TensorLayout::NCX:
          I_dims.push_back(N);
          I_dims.push_back(CI);
          for (size_t i = 0; i < spatial_rank; ++i) {
            I_dims.push_back(TensorDim(result_shape[i]));
          }
          break;
        case TensorLayout::NXC:
          I_dims.push_back(N);
          for (size_t i = 0; i < spatial_rank; ++i) {
            I_dims.push_back(TensorDim(result_shape[i]));
          }
          I_dims.push_back(CI);
          break;
        default:
          throw std::runtime_error("Invalid input_layout");
      }
      I = Tensor{name, I_dims};
      // I = NamedTensorOutput(name, I_dims);  // TODO: Re-enable when ready
      break;
    case ConvDerivMode::FILTER:
      switch (filter_layout) {
        // TODO: This won't always work for grouped convolutions, will have to update
        case TensorLayout::GKCX:
          F_dims.push_back(G);
          // Fall through deliberately
        case TensorLayout::KCX:
          F_dims.push_back(F_CO);
          F_dims.push_back(F_CI);
          for (size_t i = 0; i < spatial_rank; ++i) {
            F_dims.push_back(TensorDim(result_shape[i]));
          }
          break;
        case TensorLayout::XCK:
        case TensorLayout::XGCK:
          for (size_t i = 0; i < spatial_rank; ++i) {
            F_dims.push_back(TensorDim(result_shape[i]));
          }
          if (filter_layout == TensorLayout::XGCK) {
            F_dims.push_back(G);
          }
          F_dims.push_back(F_CI);
          F_dims.push_back(F_CO);
          break;
        default:
          throw std::runtime_error("Invalid filter_layout");
      }
      F = Tensor{name, F_dims};
      // F = NamedTensorOutput(name, F_dims);  // TODO: Re-enable when ready
      break;
    default:
      throw std::runtime_error("Invalid deriv_mode");
  }

  // Set up index formulas
  // Input data indexes
  switch (input_layout) {
    case TensorLayout::NCX:
      I_idxs.push_back(n);
      if (group_layout == GroupLayout::NONE) {
        I_idxs.push_back(ci);
      } else {
        I_idxs.push_back((CI / G) * g + ci);
      }
      for (size_t i = 0; i < spatial_rank; ++i) {
        I_idxs.emplace_back((strides[i] * x[i] + dilations[i] * k[i] - pad_before[i]) / data_dilations[i]);
      }
      break;
    case TensorLayout::NXC:
      I_idxs.push_back(n);
      for (size_t i = 0; i < spatial_rank; ++i) {
        I_idxs.emplace_back((strides[i] * x[i] + dilations[i] * k[i] - pad_before[i]) / data_dilations[i]);
      }
      if (group_layout == GroupLayout::NONE) {
        I_idxs.push_back(ci);
      } else {
        I_idxs.push_back((CI / G) * g + ci);
      }
      break;
    default:
      throw std::runtime_error("Invalid input_layout");
  }

  // Filter indexes
  TensorIndex f_co, f_ci;  // Filter index formulas for out/in channels; depend on group layout
  switch (group_layout) {
    case GroupLayout::NONE:
    case GroupLayout::SEPARATE:
      f_co = co;
      f_ci = ci;
      break;
    case GroupLayout::IN_C:
      f_co = co;
      f_ci = (CI / G) * g + ci;
      if (ci < CI / G) {
        // Do nothing; this is just a constraint
      }
      break;
    case GroupLayout::IN_K:
      f_co = (CO / G) * g + co;
      f_ci = ci;
      if (co < CO / G) {
        // Do nothing; this is just a constraint
      }
      break;
    default:
      throw std::runtime_error("Unrecognized group layout");
  }
  switch (filter_layout) {
    case TensorLayout::GKCX:
      F_idxs.push_back(g);
      // Fall through deliberately
    case TensorLayout::KCX:
      F_idxs.push_back(f_co);
      F_idxs.push_back(f_ci);
      for (size_t i = 0; i < spatial_rank; ++i) {
        F_idxs.push_back(k[i]);
      }
      break;
    case TensorLayout::XCK:
    case TensorLayout::XGCK:
      for (size_t i = 0; i < spatial_rank; ++i) {
        F_idxs.push_back(k[i]);
      }
      if (filter_layout == TensorLayout::XGCK) {
        F_idxs.push_back(g);
      }
      F_idxs.push_back(f_ci);
      F_idxs.push_back(f_co);
      break;
    default:
      throw std::runtime_error("Invalid filter_layout");
  }

  // Output data indexes
  // This assumes we infer the output layout from the input layout. So if we change that, the below switch will need to
  // be adapted.
  switch (input_layout) {
    case TensorLayout::NCX:
      O_idxs.push_back(n);
      if (group_layout == GroupLayout::NONE) {
        O_idxs.push_back(co);
      } else {
        O_idxs.push_back((CO / G) * g + co);
      }
      for (size_t i = 0; i < spatial_rank; ++i) {
        O_idxs.push_back(x[i]);
      }
      break;
    case TensorLayout::NXC:
      O_idxs.push_back(n);
      for (size_t i = 0; i < spatial_rank; ++i) {
        O_idxs.push_back(x[i]);
      }
      if (group_layout == GroupLayout::NONE) {
        O_idxs.push_back(co);
      } else {
        O_idxs.push_back((CO / G) * g + co);
      }
      break;
    default:
      throw std::runtime_error("Invalid input_layout");
  }

  // Return the contraction
  switch (deriv_mode) {
    case ConvDerivMode::NONE:
      O(O_idxs) += (I(I_idxs) * F(F_idxs));
      return Value{O};
    case ConvDerivMode::DATA:
      I(I_idxs) += (O(O_idxs) * F(F_idxs));
      return Value{I};
    case ConvDerivMode::FILTER:
      F(F_idxs) += (I(I_idxs) * O(O_idxs));
      return Value{F};
    default:
      throw std::runtime_error("Unrecognized deriv_mode");
  }
}

Value dot(const Value& value) {
  IVLOG(1, "dot");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("dot expects 2 arguments");
  }
  auto X = args[0].as_tensor();
  auto X_shape = X.shape();
  auto Y = args[1].as_tensor();
  auto Y_shape = Y.shape();
  if (X_shape.dtype() != Y_shape.dtype()) {
    throw std::runtime_error(str(boost::format("Invalid dtype in dot: X.dtype = '%1%', Y.dtype = '%2%'") %
                                 X_shape.dtype() % Y_shape.dtype()));
  }
  if (X_shape.ndims() == 1 && Y_shape.ndims() == 1) {
    TensorDim I;
    TensorIndex i;
    X.bind_dims(I);
    Y.bind_dims(I);
    auto O = TensorOutput(I);
    O(i) += X(i) * Y(i);
    return Value{O};
  }
  if (1 <= X_shape.ndims() && 2 <= Y_shape.ndims()) {
    std::vector<TensorDim> X_dims(X_shape.ndims());
    std::vector<TensorDim> Y_dims(Y_shape.ndims());
    TensorIndex z;
    std::vector<TensorIndex> X_idxs(X_shape.ndims());
    std::vector<TensorIndex> Y_idxs(Y_shape.ndims());
    X_idxs[X_shape.ndims() - 1] = z;
    Y_idxs[Y_shape.ndims() - 2] = z;
    X.bind_dims(X_dims);
    Y.bind_dims(Y_dims);
    std::vector<TensorDim> O_dims;
    std::vector<TensorIndex> O_idxs;
    for (size_t i = 0; i < X_shape.ndims() - 1; i++) {
      O_dims.push_back(X_dims[i]);
      O_idxs.push_back(X_idxs[i]);
    }
    for (size_t i = 0; i < Y_shape.ndims() - 2; i++) {
      O_dims.push_back(Y_dims[i]);
      O_idxs.push_back(Y_idxs[i]);
    }
    O_dims.push_back(Y_dims[Y_shape.ndims() - 1]);
    O_idxs.push_back(Y_idxs[Y_shape.ndims() - 1]);
    auto O = TensorOutput(O_dims);
    O(O_idxs) += X(X_idxs) * Y(Y_idxs);
    return Value{O};
  }
  throw std::runtime_error(str(boost::format("Unsupported dims for dot operation: X.dims = %1%, Y.dims = %2%") %
                               X_shape.ndims() % Y_shape.ndims()));
}

Value expand_dims(const Value& value) {
  IVLOG(1, "expand_dims");

  // Read arguments
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error(
        str(boost::format("PlaidML expand_dims op expects 2 arguments (received %1%)") % args.size()));
  }
  auto I = args[0].as_tensor();
  auto raw_axis = args[1].as_int();

  // Initialize useful values
  auto ndims = I.shape().ndims();
  // Axis is relative to _output_ tensor, which has one more dim than I
  size_t axis = normalize_axis(raw_axis, ndims + 1, "expand_dims");
  std::vector<TensorDim> I_dims(ndims);
  std::vector<TensorIndex> I_idxs;
  std::vector<TensorDim> O_dims;
  std::vector<TensorIndex> O_idxs;
  I.bind_dims(I_dims);
  for (size_t i = 0; i < ndims; ++i) {
    I_idxs.emplace_back(str(boost::format("n%1%") % i));
    if (i == axis) {
      O_dims.emplace_back(1);
      O_idxs.emplace_back("a");
    }
    O_dims.push_back(I_dims[i]);
    O_idxs.push_back(I_idxs[i]);
  }
  if (axis == ndims) {
    // This one case won't be caught in the main loop, so handle specially
    O_dims.emplace_back(1);
    O_idxs.emplace_back("a");
  }
  auto O = TensorOutput(O_dims);
  O(O_idxs) = I(I_idxs);
  return Value{O};
}

Value max(const Value& value) {
  IVLOG(1, "max");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("max expects 3 arguments");
  }
  auto I = args[0].as_tensor();
  auto I_shape = I.shape();
  auto axes = args[1];
  auto keepdims = args[2].as_bool();
  AggregationAxes agg(I_shape.ndims(), axes, keepdims);
  I.bind_dims(agg.src_dims);
  auto O = TensorOutput(agg.dst_dims);
  O(agg.dst_idxs) >= I(agg.src_idxs);
  return Value{O};
}

Value mean(const Value& value) {
  IVLOG(1, "mean");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("mean expects 3 arguments");
  }

  auto I = args[0].as_tensor();
  auto I_shape = I.shape();
  if (I_shape.ndims() == 0) {
    return Value{I};
  }

  // TODO: Move this commented block to Keras?
  // if (I_shape.dtype() == PLAIDML_DATA_BOOLEAN) {
  //   I = cast(I, floatx());
  // }

  auto axes = args[1];
  if (axes.is_tuple() && axes.as_tuple().empty()) {
    return Value{I};
  }

  auto keepdims = args[2].as_int();

  AggregationAxes agg(I_shape.ndims(), axes, keepdims);

  I.bind_dims(agg.src_dims);
  auto SO = TensorOutput(agg.dst_dims);
  SO(agg.dst_idxs) += I(agg.src_idxs);
  auto denom = Tensor{1};
  for (const auto& axis : agg.axes) {
    denom = denom * agg.src_dims.at(axis);
  }
  return Value{SO / denom};
}

Value min(const Value& value) {
  IVLOG(1, "min");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("min expects 3 arguments");
  }
  auto I = args[0].as_tensor();
  auto I_shape = I.shape();
  auto axes = args[1];
  auto keepdims = args[2].as_bool();
  AggregationAxes agg(I_shape.ndims(), axes, keepdims);
  I.bind_dims(agg.src_dims);
  auto O = TensorOutput(agg.dst_dims);
  O(agg.dst_idxs) <= I(agg.src_idxs);
  return Value{O};
}

Value pool(const Value& value) {
  // The parameters of pool:
  //    0. Input Tensor
  //    1. Pool Mode (avg/max)
  //    2. Pool Size
  //    3. Strides
  //    4. Autopad Mode (explicit, same_lower, same_upper, valid, [maybe full?])
  //    5. Manual Padding
  //    6. Layout (i.e. Channel Order) (minimally NXC v NCX)
  //    7. Include Padding in Avg Computation (bool)
  //    8. Ceil Mode (i.e. as in ONNX)
  // n.b. We determine the number of spatial dimensions from the Pool Size and confirm it is consistent with other
  // parameters that imply a spatial dimension size, specifically strides. We do also check this against the input
  // tensor shape and the manual padding, but these are less strict: manual padding may omit some padding values
  // (which are then assumed to be 0), and the input tensor shape may have multiple channel dimensions (i.e. for
  // cases like tensors going into or coming out of grouped convolutions).

  // Read arguments
  auto args = value.as_tuple();
  if (args.size() != 9) {
    throw std::runtime_error(str(boost::format("PlaidML pool op expects 9 arguments (received %1%)") % args.size()));
  }
  auto I = args[0].as_tensor();
  auto pool_mode = pool_mode_from_str(args[1].as_str());
  auto pool_size = args[2].as_int_tuple();
  auto strides = args[3].as_int_tuple();
  auto autopad_mode = autopad_mode_from_str(args[4].as_str());
  auto manual_padding = args[5].as_int_tuple();
  auto input_layout = tensor_layout_from_str(args[6].as_str());
  auto include_padding_in_avg = args[7].as_bool();
  auto use_ceil_for_output_shape = args[8].as_bool();

  // Initialize useful values
  auto spatial_rank = pool_size.size();
  auto I_shape = I.shape();
  auto I_channel_dims = I_shape.ndims() - spatial_rank - 1;

  // Verify inputs are consistent
  if (manual_padding.size() && autopad_mode != AutopadMode::NONE) {
    throw std::runtime_error("Autopadding and manual padding both requested for single pool operation");
  }
  if (strides.size() != spatial_rank) {
    throw std::runtime_error(
        str(boost::format("Inconsistent spatial rank in pool op (received %1%D pool_size and %2%D strides)") %
            spatial_rank % strides.size()));
  }
  if (I_channel_dims != 1) {
    throw std::runtime_error(
        str(boost::format("Inconsistent spatial rank in pool op (pool_size has %1% spatial dimensions but input tensor "
                          "has %2% dimensions, and thus %3% spatial dims)") %
            spatial_rank % I.shape().ndims() % (I.shape().ndims() - 2)));
  }
  if (!is_input_layout(input_layout)) {
    throw std::runtime_error("Tensor layout requested in pool op does not apply to pooling");
  }

  extend_manual_padding(&manual_padding, spatial_rank);

  TensorDim N, C;
  TensorIndex n, c;
  std::vector<TensorDim> X(spatial_rank);
  std::vector<TensorIndex> x(spatial_rank);
  std::vector<TensorIndex> k(spatial_rank);  // within-pool spatial indexes

  std::vector<TensorDim> pad_before;
  std::vector<TensorDim> I_dims = {N};
  std::vector<TensorIndex> I_idxs = {n};
  std::vector<TensorDim> O_dims = {N};
  std::vector<TensorIndex> O_idxs = {n};
  if (input_layout == TensorLayout::NCX) {
    I_dims.push_back(C);
  }
  for (size_t i = 0; i < spatial_rank; ++i) {
    I_dims.push_back(X[i]);
  }
  if (input_layout == TensorLayout::NXC) {
    I_dims.push_back(C);
  }
  I.bind_dims(I_dims);
  if (input_layout == TensorLayout::NCX) {
    I_idxs.push_back(c);
    O_dims.push_back(C);
    O_idxs.push_back(c);
  }
  for (size_t i = 0; i < spatial_rank; ++i) {
    O_idxs.push_back(x[i]);
    TensorDim local_pad_before;
    TensorDim local_output_size;
    TensorIndex local_index;
    std::tie(local_pad_before, local_output_size) =
        compute_padding_and_output_size(X[i], TensorDim(pool_size[i]), strides[i], autopad_mode, manual_padding[i],
                                        manual_padding[spatial_rank + i], 1, 1, use_ceil_for_output_shape);
    pad_before.emplace_back(local_pad_before);
    local_index = strides[i] * x[i] + k[i] - pad_before[i];
    O_dims.emplace_back(local_output_size);
    I_idxs.emplace_back(local_index);
    // Add constraints: the format is weird but this does actually apply the constraints
    if (k[i] < pool_size[i]) {
      // nolint(whitespace/empty_if_body)
    }
  }
  if (input_layout == TensorLayout::NXC) {
    I_idxs.push_back(c);
    O_dims.push_back(C);
    O_idxs.push_back(c);
  }
  auto O = TensorOutput(O_dims);
  if (pool_mode == PoolMode::MAX) {
    O(O_idxs) >= I(I_idxs);
    return Value{O};
  } else if (pool_mode == PoolMode::MIN) {
    O(O_idxs) <= I(I_idxs);
    return Value{O};
  } else if (pool_mode == PoolMode::SUM) {
    O(O_idxs) += I(I_idxs);
    return Value{O};
  } else if (pool_mode == PoolMode::AVG) {
    O(O_idxs) += I(I_idxs);
    if (include_padding_in_avg) {
      int64_t total_pool_size = 1;
      for (const auto& sz : pool_size) {
        total_pool_size *= sz;
      }
      return Value{O / total_pool_size};
    } else {
      auto One = Tensor{1};
      auto Ones = TensorOutput(I_dims);
      auto Count = TensorOutput(O_dims);
      // Note: O_idxs is used in both cases b/c both need indexes of the form x0, x1, ...
      // However, they do not represent the same index values (and notably do not interate
      // over the same size of dimensions as I_dims != O_dims)
      Ones(O_idxs) = One(std::vector<TensorIndex>());
      Count(O_idxs) += Ones(I_idxs);
      return Value{O / Count};
    }
  } else {
    throw std::runtime_error("Unrecognized pool_mode in pool op");
  }
}

Value relu(const Value& value) {
  IVLOG(1, "relu");
  auto args = value.as_tuple();
  if (args.size() != 4) {
    throw std::runtime_error("relu expects 4 arguments");
  }
  auto I = args[0].as_tensor();
  auto alpha = args[1];
  auto max_value = args[2];
  auto threshold = args[3].as_float();
  Tensor A;
  if (alpha.is_none()) {
    A = Tensor(0.0);
  } else {
    A = alpha.as_tensor();
  }
  auto O = select(I < threshold, A * (I - threshold), I);
  if (!max_value.is_none()) {
    auto M = max_value.as_tensor();
    O = select(O < M, O, M);
  }
  return Value{O};
}

Value softmax(const Value& value) {
  IVLOG(1, "softmax");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("softmax expects 2 arguments");
  }
  auto X = args[0].as_tensor();
  if (X.shape().ndims() == 2) {
    TensorDim I, J;
    TensorIndex i, j;
    X.bind_dims(I, J);
    auto M = TensorOutput(I, 1);
    M(i, 0) >= X(i, j);
    auto E = exp(X - M);
    auto N = TensorOutput(I, 1);
    N(i, 0) += E(i, j);
    return Value{E / N};
  }
  throw std::runtime_error("softmax only works on 2 dimensions at this time.");
}

Value spatial_padding(const Value& value) {
  IVLOG(1, "spatial_padding");
  auto args = value.as_tuple();
  if (args.size() != 4) {
    throw std::runtime_error("spatial_padding expects 4 arguments");
  }
  auto I = args[0].as_tensor();
  auto lo_pads = args[1].as_int_tuple();
  auto hi_pads = args[2].as_int_tuple();
  auto data_layout = tensor_layout_from_str(args[3].as_str());

  // validate inputs
  auto nonspatial_ndims = nonspatial_dims(data_layout);
  auto spatial_rank = I.shape().ndims() - nonspatial_ndims;

  if (spatial_rank < 1) {
    throw std::runtime_error(str(
        boost::format(
            "Insufficient spatial rank in spatial_padding op (At least 1 spatial dim required; received %1% spatial "
            "dims based on an input tensor with %2% dims with a specified layout with %3% nonspatial dims.") %
        spatial_rank % I.shape().ndims() % nonspatial_ndims));
  }
  if (lo_pads.size() != spatial_rank) {
    throw std::runtime_error(
        str(boost::format("Inconsistent spatial rank in spatial_padding op (received %1% spatial dim(s) based on an "
                          "input tensor with %2% dims with a specified layout with %3% nonspatial dims, but received "
                          "lower padding for %4% spatial dims.") %
            spatial_rank % I.shape().ndims() % nonspatial_ndims % lo_pads.size()));
  }
  if (hi_pads.size() != spatial_rank) {
    throw std::runtime_error(
        str(boost::format("Inconsistent spatial rank in spatial_padding op (received %1% spatial dim(s) based on an "
                          "input tensor with %2% dims with a specified layout with %3% nonspatial dims, but received "
                          "upper padding for %4% spatial dims.") %
            spatial_rank % I.shape().ndims() % nonspatial_ndims % hi_pads.size()));
  }

  std::vector<TensorDim> I_dims;
  std::vector<TensorDim> O_dims;
  std::vector<TensorIndex> I_idxs;
  std::vector<TensorIndex> O_idxs;

  switch (data_layout) {
    case TensorLayout::GKCX: {
      IVLOG(1, "Spatial padding requested for tensor with kernel-style layout.");
      // Initialize dims & indexes
      TensorDim G, K, C;
      TensorIndex g("g");
      TensorIndex k("k");
      TensorIndex c("c");
      std::vector<TensorDim> X(spatial_rank);
      std::vector<TensorIndex> x;
      for (size_t i = 0; i < spatial_rank; ++i) {
        x.emplace_back(str(boost::format("x%1%") % i));
      }

      // Assign input dims & indexes
      I_dims.push_back(G);
      I_idxs.push_back(g);
      I_dims.push_back(K);
      I_idxs.push_back(k);
      I_dims.push_back(C);
      I_idxs.push_back(c);
      for (size_t i = 0; i < spatial_rank; ++i) {
        I_dims.push_back(X[i]);
        I_idxs.push_back(x[i]);
      }
      I.bind_dims(I_dims);

      // Assign output dims & indexes
      O_dims.push_back(G);
      O_idxs.push_back(g);
      O_dims.push_back(K);
      O_idxs.push_back(k);
      O_dims.push_back(C);
      O_idxs.push_back(c);
      for (size_t i = 0; i < spatial_rank; ++i) {
        O_dims.push_back(X[i] + lo_pads[i] + hi_pads[i]);
        O_idxs.push_back(x[i] + lo_pads[i]);
      }
    } break;
    case TensorLayout::KCX: {
      IVLOG(1, "Spatial padding requested for tensor with kernel-style layout.");
      // Initialize dims & indexes
      TensorDim K, C;
      TensorIndex k("k");
      TensorIndex c("c");
      std::vector<TensorDim> X(spatial_rank);
      std::vector<TensorIndex> x;
      for (size_t i = 0; i < spatial_rank; ++i) {
        x.emplace_back(str(boost::format("x%1%") % i));
      }

      // Assign input dims & indexes
      I_dims.push_back(K);
      I_idxs.push_back(k);
      I_dims.push_back(C);
      I_idxs.push_back(c);
      for (size_t i = 0; i < spatial_rank; ++i) {
        I_dims.push_back(X[i]);
        I_idxs.push_back(x[i]);
      }
      I.bind_dims(I_dims);

      // Assign output dims & indexes
      O_dims.push_back(K);
      O_idxs.push_back(k);
      O_dims.push_back(C);
      O_idxs.push_back(c);
      for (size_t i = 0; i < spatial_rank; ++i) {
        O_dims.push_back(X[i] + lo_pads[i] + hi_pads[i]);
        O_idxs.push_back(x[i] + lo_pads[i]);
      }
    } break;
    case TensorLayout::NCX: {
      TensorDim N, C;
      TensorIndex n("n");
      TensorIndex c("c");
      std::vector<TensorDim> X(spatial_rank);
      std::vector<TensorIndex> x;
      for (size_t i = 0; i < spatial_rank; ++i) {
        x.emplace_back(str(boost::format("x%1%") % i));
      }

      // Assign input dims & indexes
      I_dims.push_back(N);
      I_idxs.push_back(n);
      I_dims.push_back(C);
      I_idxs.push_back(c);
      for (size_t i = 0; i < spatial_rank; ++i) {
        I_dims.push_back(X[i]);
        I_idxs.push_back(x[i]);
      }
      I.bind_dims(I_dims);

      // Assign output dims & indexes
      O_dims.push_back(N);
      O_idxs.push_back(n);
      O_dims.push_back(C);
      O_idxs.push_back(c);
      for (size_t i = 0; i < spatial_rank; ++i) {
        O_dims.push_back(X[i] + lo_pads[i] + hi_pads[i]);
        O_idxs.push_back(x[i] + lo_pads[i]);
      }
    } break;
    case TensorLayout::NXC: {
      TensorDim N, C;
      TensorIndex n("n");
      TensorIndex c("c");
      std::vector<TensorDim> X(spatial_rank);
      std::vector<TensorIndex> x;
      for (size_t i = 0; i < spatial_rank; ++i) {
        x.emplace_back(str(boost::format("x%1%") % i));
      }

      // Assign input dims & indexes
      I_dims.push_back(N);
      I_idxs.push_back(n);
      for (size_t i = 0; i < spatial_rank; ++i) {
        I_dims.push_back(X[i]);
        I_idxs.push_back(x[i]);
      }
      I_dims.push_back(C);
      I_idxs.push_back(c);
      I.bind_dims(I_dims);

      // Assign output dims & indexes
      O_dims.push_back(N);
      O_idxs.push_back(n);
      for (size_t i = 0; i < spatial_rank; ++i) {
        O_dims.push_back(X[i] + lo_pads[i] + hi_pads[i]);
        O_idxs.push_back(x[i] + lo_pads[i]);
      }
      O_dims.push_back(C);
      O_idxs.push_back(c);
    } break;
    case TensorLayout::XCK: {
      IVLOG(1, "Spatial padding requested for tensor with kernel-style layout.");
      TensorDim C, K;
      TensorIndex c("c");
      TensorIndex k("k");
      std::vector<TensorDim> X(spatial_rank);
      std::vector<TensorIndex> x;
      for (size_t i = 0; i < spatial_rank; ++i) {
        x.emplace_back(str(boost::format("x%1%") % i));
      }

      // Assign input dims & indexes
      for (size_t i = 0; i < spatial_rank; ++i) {
        I_dims.push_back(X[i]);
        I_idxs.push_back(x[i]);
      }
      I_dims.push_back(C);
      I_idxs.push_back(c);
      I_dims.push_back(K);
      I_idxs.push_back(k);
      I.bind_dims(I_dims);

      // Assign output dims & indexes
      for (size_t i = 0; i < spatial_rank; ++i) {
        O_dims.push_back(X[i] + lo_pads[i] + hi_pads[i]);
        O_idxs.push_back(x[i] + lo_pads[i]);
      }
      O_dims.push_back(C);
      O_idxs.push_back(c);
      O_dims.push_back(K);
      O_idxs.push_back(k);
    } break;
    case TensorLayout::XGCK: {
      IVLOG(1, "Spatial padding requested for tensor with kernel-style layout.");
      TensorDim G, C, K;
      TensorIndex g("g");
      TensorIndex c("c");
      TensorIndex k("k");
      std::vector<TensorDim> X(spatial_rank);
      std::vector<TensorIndex> x;
      for (size_t i = 0; i < spatial_rank; ++i) {
        x.emplace_back(str(boost::format("x%1%") % i));
      }

      // Assign input dims & indexes
      for (size_t i = 0; i < spatial_rank; ++i) {
        I_dims.push_back(X[i]);
        I_idxs.push_back(x[i]);
      }
      I_dims.push_back(G);
      I_idxs.push_back(g);
      I_dims.push_back(C);
      I_idxs.push_back(c);
      I_dims.push_back(K);
      I_idxs.push_back(k);
      I.bind_dims(I_dims);

      // Assign output dims & indexes
      for (size_t i = 0; i < spatial_rank; ++i) {
        O_dims.push_back(X[i] + lo_pads[i] + hi_pads[i]);
        O_idxs.push_back(x[i] + lo_pads[i]);
      }
      O_dims.push_back(G);
      O_idxs.push_back(g);
      O_dims.push_back(C);
      O_idxs.push_back(c);
      O_dims.push_back(K);
      O_idxs.push_back(k);
    } break;
    default:
      throw std::runtime_error("Unrecognized TensorLayout in spatial_padding");
  }
  auto O = TensorOutput(O_dims);
  O(O_idxs) = I(I_idxs);
  return Value{O};
}

Value square(const Value& value) {
  IVLOG(1, "square");
  auto x = value.as_tensor();
  return Value(x * x);
}

Value squeeze(const Value& value) {
  IVLOG(1, "squeeze");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("Squeeze expects 2 arguments");
  }

  // Read arguments
  auto I = args[0].as_tensor();
  auto raw_axis = args[1].as_int();

  // Validate arguments
  auto ndims = I.shape().ndims();
  auto axis = normalize_axis(raw_axis, ndims, "squeeze");
  std::vector<TensorDim> I_dims(ndims);
  std::vector<TensorIndex> I_idxs;
  std::vector<TensorDim> O_dims;
  std::vector<TensorIndex> O_idxs;
  I.bind_dims(I_dims);
  for (size_t i = 0; i < ndims; ++i) {
    I_idxs.emplace_back(str(boost::format("n%1%") % i));
    if (i != axis) {
      O_dims.push_back(I_dims[i]);
      O_idxs.push_back(I_idxs[i]);
    }
  }
  auto O = TensorOutput(O_dims);
  O(O_idxs) = I(I_idxs);
  return Value{O};
}

Value sum(const Value& value) {
  IVLOG(1, "sum");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("mean expects 3 arguments");
  }

  auto I = args[0].as_tensor();
  auto I_shape = I.shape();
  if (I_shape.ndims() == 0) {
    return Value{I};
  }

  // TODO: Move this commented block to Keras?
  // if (I_shape.dtype() == PLAIDML_DATA_BOOLEAN) {
  //   I = cast(I, floatx());
  // }

  auto axes = args[1];
  if (axes.is_tuple() && axes.as_tuple().empty()) {
    return Value{I};
  }

  auto keepdims = args[2].as_int();

  AggregationAxes agg(I_shape.ndims(), axes, keepdims);

  I.bind_dims(agg.src_dims);
  auto O = TensorOutput(agg.dst_dims);
  O(agg.dst_idxs) += I(agg.src_idxs);

  return Value{O};
}

Value tile(const Value& value) {
  IVLOG(1, "tile");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("Tile expects 2 arguments");
  }

  // Read arguments
  auto I = args[0].as_tensor();
  auto reps = args[1].as_int_tuple();

  // Validate args
  auto ndims = I.shape().ndims();
  if (reps.empty()) {
    throw std::runtime_error("No tiling factors provided to tile operation");
  }
  while (reps.size() < ndims) {
    // Numpy style: extend to full rank with 1s
    reps.push_back(1);
  }
  if (reps.size() > ndims) {
    throw std::runtime_error("More tiling factors provided to tile operation than tensor dimensions");
  }

  std::vector<TensorDim> I_dims(ndims);
  std::vector<TensorIndex> I_idxs(ndims);
  I.bind_dims(I_dims);
  std::vector<TensorDim> O_dims;
  std::vector<TensorIndex> O_idxs;
  for (size_t i = 0; i < ndims; ++i) {
    O_dims.push_back(I_dims[i] * reps[i]);
    O_idxs.push_back(TensorIndex() * I_dims[i] + I_idxs[i]);
  }
  auto O = TensorOutput(O_dims);
  O(O_idxs) = I(I_idxs);
  O.no_defract();
  return Value{O};
}

Value transpose(const Value& value) {
  // Reorders dimensions so dim i of the output is dim pattern[i] of the input
  IVLOG(1, "transpose");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("Transpose expects 2 arguments");
  }

  // Read arguments
  auto I = args[0].as_tensor();
  auto pattern_val = args[1];

  // Normalize pattern value
  auto I_shape = I.shape();
  auto ndims = I_shape.ndims();
  std::vector<int64_t> pattern;
  if (pattern_val.is_none()) {
    // Default is to reverse the dimensions
    for (size_t i = 0; i < ndims; ++i) {
      pattern.push_back(ndims - 1 - i);
    }
  } else if (pattern_val.is_tuple()) {
    pattern = pattern_val.as_int_tuple();
  } else {
    throw std::runtime_error("Transpose 2nd argument must be none or integer tuple");
  }

  // Ensure pattern is sane
  for (const auto& i : pattern) {
    if (i < 0 || ndims <= static_cast<size_t>(i)) {
      throw std::runtime_error(
          str(boost::format("Transpose of nonexistent axis %1% requested (input has %2% dimensions)") % i % ndims));
    }
  }

  // Setup inputs and outputs
  std::vector<TensorDim> I_dims(ndims);
  std::vector<TensorIndex> I_idxs(ndims);
  I.bind_dims(I_dims);
  std::vector<TensorDim> O_dims;
  std::vector<TensorIndex> O_idxs;
  for (size_t i = 0; i < ndims; ++i) {
    O_dims.push_back(I_dims[pattern[i]]);
    O_idxs.push_back(I_idxs[pattern[i]]);
  }
  auto O = TensorOutput(O_dims);
  O(O_idxs) = I(I_idxs);
  return Value{O};
}

Value variance(const Value& value) {
  // This computes the *uncorrected* sample variance (i.e. denominator = n rather than = n-1) to match tensorflow
  IVLOG(1, "variance");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("Variance expects 3 arguments");
  }

  // Read arguments
  auto I = args[0].as_tensor();
  auto axes = args[1];
  auto keepdims = args[2].as_int();

  // Handle trivial cases
  if (I.shape().ndims() == 0) {
    // TODO: Adjust for dtype?
    return Value{0};
  }
  if (axes.is_tuple() && axes.as_tuple().empty()) {
    throw std::runtime_error("Variance expects nonempty axis list");
  }

  // TODO: Move this commented block to Keras?
  // if (I.shape().dtype() == PLAIDML_DATA_BOOLEAN) {
  //   I = cast(I, floatx());
  // }

  auto Mean = mean(edsl::make_tuple(I, axes, true)).as_tensor();
  AggregationAxes agg(I.shape().ndims(), axes, keepdims);

  I.bind_dims(agg.src_dims);

  auto SquaredDifference = (I - Mean) * (I - Mean);
  auto SumSqDiff = TensorOutput(agg.dst_dims);
  SumSqDiff(agg.dst_idxs) += SquaredDifference(agg.src_idxs);
  auto denom = Tensor{1};
  for (const auto& axis : agg.axes) {
    denom = denom * agg.src_dims.at(axis);
  }
  return Value{SumSqDiff / denom};
}

void RegisterOps() {
  auto registry = OperationRegistry::Instance();
  registry->Register("abs", abs);
  registry->Register("argmax", argmax);
  registry->Register("concatenate", concatenate);
  registry->Register("convolution", convolution);
  registry->Register("dot", dot);
  registry->Register("expand_dims", expand_dims);
  registry->Register("min", min);
  registry->Register("mean", mean);
  registry->Register("max", max);
  registry->Register("pool", pool);
  registry->Register("relu", relu);
  registry->Register("softmax", softmax);
  registry->Register("spatial_padding", spatial_padding);
  registry->Register("square", square);
  registry->Register("squeeze", squeeze);
  registry->Register("sum", sum);
  registry->Register("tile", tile);
  registry->Register("transpose", transpose);
  registry->Register("variance", variance);
}

}  // namespace lib
}  // namespace op
}  // namespace plaidml
