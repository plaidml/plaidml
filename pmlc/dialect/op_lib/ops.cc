// Copyright 2019 Intel Corporation.

#include "pmlc/dialect/op_lib/op_lib_wrappers.h.inc"
#include "plaidml2/op/lib/ops.h"

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

#include <boost/format.hpp>

#include "base/util/logging.h"

using namespace plaidml::edsl;  // NOLINT
using namespace plaidml::op;    // NOLINT

namespace plaidml::op::lib {

// Forward declare the operations here so they can call each other
Value abs(const Value&);
Value convolution(const Value&);

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

enum class AutoDimMode {
  MATCH,  // 0
  FILL    // -1
};

namespace {
// TODO: I haven't decided whether to make these helper functions visible to the outside world

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

std::pair<TensorDim, TensorDim> compute_padding_and_output_size(  //
    const TensorDim& input_size,                                  //
    const TensorDim& filter_size,                                 //
    int64_t stride,                                               //
    AutopadMode autopad_mode,                                     //
    int64_t pad_lo,                                               //
    int64_t pad_hi,                                               //
    int64_t dilation,                                             //
    int64_t data_dilation,                                        //
    bool use_ceil_for_output_shape) {
  // Effective input and filter sizes are the sizes after dilations are
  // accounted for. So a 4x3 filter dilated by (3, 2) has an effective filter
  // size of 11 and 5 for its 2 spatial dims

  auto I_eff = (data_dilation * (input_size - 1)) + 1;  // Effective Input Size
  auto F_eff = (dilation * (filter_size - 1)) + 1;      // Effective Filter Size
  int64_t ceil_term =
      use_ceil_for_output_shape ? stride - 1 : 0;  // TODO: Will need to confirm that this is the intended behavior
  if (autopad_mode == AutopadMode::NONE) {
    TensorDim pad_before(pad_lo);
    TensorDim output_size((I_eff + pad_lo + pad_hi - F_eff + stride + ceil_term) / stride);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  } else if (autopad_mode == AutopadMode::VALID) {
    TensorDim pad_before(0);
    TensorDim output_size((I_eff - F_eff + stride + ceil_term) / stride);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  } else if (autopad_mode == AutopadMode::SAME_LOWER || autopad_mode == AutopadMode::SAME_UPPER) {
    TensorDim output_size((I_eff + stride - 1 + ceil_term) / stride);
    int64_t lower_term = (autopad_mode == AutopadMode::SAME_LOWER) ? 1 : 0;
    TensorDim pad_before((max(0, (output_size - 1) * stride + F_eff - I_eff) + lower_term) / 2);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  } else {
    throw std::runtime_error("invalid autopad mode");
  }
  // TODO: add back in runtime error with string formatting that was removed for POC purposes
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
  AutopadMode autopad_mode = static_cast<AutopadMode>(args[7].as_int());
  auto manual_padding = args[8].as_int_tuple();
  TensorLayout input_layout = static_cast<TensorLayout>(args[9].as_int());
  TensorLayout filter_layout = static_cast<TensorLayout>(args[10].as_int());
  GroupLayout group_layout = static_cast<GroupLayout>(args[11].as_int());
  // auto winograd_allowed = args[12].as_bool();  // TODO: Implement Winograd
  auto name = args[13].as_str();
  AutogroupMode autogroup_mode = static_cast<AutogroupMode>(args[14].as_int());
  DerivMode deriv_mode = static_cast<DerivMode>(args[15].as_int());
  auto result_shape = args[16].as_int_tuple();

  Tensor I;       // Inputs (i.e. Data) tensor
  Tensor F;       // Filters (i.e. Weights i.e. Kernel) tensor
  Tensor O;       // Output (i.e. of a forward pass) tensor
  Tensor Result;  // The tensor that is actually returned, depends on deriv_mode

  // Connect the inputs to the right names
  switch (deriv_mode) {
    case DerivMode::NONE:
      I = I_or_O;
      F = F_or_O;
      break;
    case DerivMode::DATA:
      O = I_or_O;
      F = F_or_O;
      break;
    case DerivMode::FILTER:
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
  if (deriv_mode != DerivMode::DATA && I.shape().ndims() - spatial_rank != nonspatial_dims(input_layout)) {
    // If we ever extend possible layouts so that I and O may have different layouts, we will
    // need to do this check in different ways depending on whether deriv_mode is DATA or not
    throw std::runtime_error(
        str(boost::format("Inconsistent spatial rank in conv op (received %1% spatial dimensions based on strides but "
                          "input tensor has %2% dimensions, and thus %3% spatial dims). (This error can also occur if "
                          "the layout of I is incorrectly specified or interpreted.)") %
            spatial_rank % I.shape().ndims() % (I.shape().ndims() - nonspatial_dims(input_layout))));
  }
  if (deriv_mode != DerivMode::FILTER && F.shape().ndims() - spatial_rank != nonspatial_dims(filter_layout)) {
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
    if (deriv_mode != DerivMode::NONE) {
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
  // The channel dimensions as used by the filters, adjusted for group layout
  TensorDim F_CI, F_CO;
  TensorIndex n("n");
  TensorIndex ci("ci");
  TensorIndex co("co");
  TensorIndex g("g");
  // The spatial dimensions of I
  std::vector<TensorDim> X(spatial_rank);
  // The spatial indexes of I
  std::vector<TensorIndex> x;
  for (size_t i = 0; i < spatial_rank; ++i) {
    x.emplace_back(TensorIndex(str(boost::format("x%1%") % i)));
  }
  // The spatial dimensions of O; nearly unused
  std::vector<TensorDim> Y(spatial_rank);
  // The spatial dimensions of F
  std::vector<TensorDim> K(spatial_rank);
  // The spatial indexs of F
  std::vector<TensorIndex> k;
  for (size_t i = 0; i < spatial_rank; ++i) {
    k.emplace_back(TensorIndex(str(boost::format("k%1%") % i)));
  }
  std::vector<TensorDim> I_dims;
  std::vector<TensorIndex> I_idxs;
  std::vector<TensorDim> F_dims;
  // this ensures that the inferred filter shape matches filter_shape if the latter is passed in
  std::vector<TensorDim> F_explicit_dims;
  std::vector<TensorIndex> F_idxs;
  std::vector<TensorDim> O_dims;
  std::vector<TensorIndex> O_idxs;
  // G may be explicit or automatically set, based on autogroup_mode
  TensorDim G_explicit(groups);
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
        // TODO: add back in runtime error with string formatting that was removed for POC purposes
      }
      break;
    case AutogroupMode::AUTO:
      if (group_layout == GroupLayout::SEPARATE || group_layout == GroupLayout::IN_K) {
        // just let G be inferred; i.e. do nothing  // nolint(whitespace/empty_if_body)
      } else {
        // TODO: add back in runtime error with string formatting that was removed for POC purposes
      }
      break;
    default:
      throw std::runtime_error("Unrecognized AutogroupMode");
  }

  // Set up dimensions of the inputs first so they can be bound Group layout
  // affects the size of filter dimensions; we pass through the dims that don't
  // need to be adjusted here, and we will calculate those dimensions that will
  // be adjusted later (after some more dims are bound).
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
  if (deriv_mode != DerivMode::DATA) {
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
  if (deriv_mode != DerivMode::FILTER) {
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
  if (deriv_mode != DerivMode::NONE) {
    // This assumes we infer the output layout from the input layout. So if we
    // change that, the output data dims section will need to be adapted.
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
    TensorDim local_input_size = (deriv_mode == DerivMode::DATA) ? TensorDim(result_shape[i]) : X[i];
    TensorDim local_filter_size = (deriv_mode == DerivMode::FILTER) ? TensorDim(result_shape[i]) : K[i];
    std::tie(local_pad_before, local_output_size) = compute_padding_and_output_size(
        local_input_size, local_filter_size, strides[i], autopad_mode, manual_padding[i],
        manual_padding[i + spatial_rank], dilations[i], data_dilations[i], false);
    pad_before.emplace_back(local_pad_before);
    O_spatial_dims.emplace_back(local_output_size);
  }

  // Now set up the dimensions of the result to be returned
  switch (deriv_mode) {
    case DerivMode::NONE:
      // This assumes we infer the output layout from the input layout. So if we
      // change that, the below switch will need to be adapted.
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
    case DerivMode::DATA:
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
    case DerivMode::FILTER:
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

  std::vector<Constraint> constraints;

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
      constraints.push_back(ci < CI / G);
      break;
    case GroupLayout::IN_K:
      f_co = (CO / G) * g + co;
      f_ci = ci;
      constraints.push_back(co < CO / G);
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
  // This assumes we infer the output layout from the input layout. So if we
  // change that, the below switch will need to be adapted.
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
    case DerivMode::NONE:
      O(O_idxs) += I(I_idxs) * F(F_idxs);
      O.add_constraints(constraints);
      return Value{O};
    case DerivMode::DATA:
      I(I_idxs) += O(O_idxs) * F(F_idxs);
      I.add_constraints(constraints);
      return Value{I};
    case DerivMode::FILTER:
      F(F_idxs) += I(I_idxs) * O(O_idxs);
      F.add_constraints(constraints);
      return Value{F};
    default:
      throw std::runtime_error("Unrecognized deriv_mode");
  }
}

void RegisterOps() {
  auto registry = OperationRegistry::Instance();
  registry->Register("abs", abs);
  registry->Register("convolution", convolution);
}

}  // namespace plaidml::op::lib
