// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace {

template <typename T>
std::vector<T> cast_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto* ngraph_const = ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(operand_idx));
  if (ngraph_const) {
    return ngraph_const->cast_vector<T>();
  } else {
    THROW_IE_EXCEPTION
        << "Dynamic slicing not currently supported by PlaidML plugin; all of indices, offsets and default index"
           "must be Constants.";
  }
}

}  // namespace

namespace PlaidMLPlugin {

edsl::InterpolationMode get_plaidml_interpolation_mode(ngraph::op::v4::Interpolate::InterpolateMode mode) {
  switch (mode) {
    case ngraph::op::v4::Interpolate::InterpolateMode::nearest:
      return edsl::InterpolationMode::NEAREST;
    case ngraph::op::v4::Interpolate::InterpolateMode::linear:
      return edsl::InterpolationMode::LINEAR;
    case ngraph::op::v4::Interpolate::InterpolateMode::cubic:
      return edsl::InterpolationMode::CUBIC;
    default:
      THROW_IE_EXCEPTION << "Unsupported Interpolate InterpolateMode";
      break;
  }
}

edsl::NearestMode get_plaidml_nearest_mode(ngraph::op::v4::Interpolate::NearestMode nearest_mode, bool is_downsample) {
  switch (nearest_mode) {
    case ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor:
      return edsl::NearestMode::ROUND_PREFER_FLOOR;
    case ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil:
      return edsl::NearestMode::ROUND_PREFER_CEIL;
    case ngraph::op::v4::Interpolate::NearestMode::floor:
      return edsl::NearestMode::FLOOR;
    case ngraph::op::v4::Interpolate::NearestMode::ceil:
      return edsl::NearestMode::CEIL;
    case ngraph::op::v4::Interpolate::NearestMode::simple:
      return is_downsample ? edsl::NearestMode::CEIL : edsl::NearestMode::SIMPLE;
    default:
      THROW_IE_EXCEPTION << "Unsupported Interpolate NearestMode";
      break;
  }
}

edsl::Tensor get_coordinate_transformed_indices(
    int64_t I_dim, float scale, ngraph::op::v4::Interpolate::CoordinateTransformMode coordinate_transformation_mode) {
  int64_t O_dim = floor(I_dim * scale);
  auto IX = edsl::cast(edsl::index({edsl::TensorDim(O_dim)}, 0), DType::FLOAT32);
  if (scale == 1.0 || (O_dim == I_dim)) {
    return IX;
  }
  switch (coordinate_transformation_mode) {
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn:
      IX = (IX + 0.5) / scale;
      break;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel:
      if (O_dim > 1) {
        IX = (IX + 0.5) / scale - 0.5;
      }
      break;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel:
      IX = (IX + 0.5) / scale - 0.5;
      break;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric:
      IX = IX / scale;
      break;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners:
      if (O_dim > 1) {
        IX = IX * (I_dim - 1) / (O_dim - 1);
      }
      break;
    default:
      THROW_IE_EXCEPTION << "Unsupported Interpolate CoordinateTransformMode";
      break;
  }
  return IX;
}

void registerInterpolate() {
  registerOp("Interpolate", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::Interpolate>(ctx.layer);

    // Inputs
    auto I = ctx.operands.at(0);
    auto result_shape = cast_constant_operand<int64_t>(1, layer);
    auto default_scales = cast_constant_operand<float>(2, layer);
    auto axes = cast_constant_operand<int64_t>(3, layer);

    // Attributes
    auto mode = layer->get_attrs().mode;
    auto shape_calculation_mode = layer->get_attrs().shape_calculation_mode;
    auto coordinate_transformation_mode = layer->get_attrs().coordinate_transformation_mode;
    auto nearest_mode = layer->get_attrs().nearest_mode;
    auto pads_begin = layer->get_attrs().pads_begin;
    auto pads_end = layer->get_attrs().pads_end;
    auto cube_coeff = layer->get_attrs().cube_coeff;

    // Padding
    I = op::explicit_padding(I, {pads_begin.begin(), pads_begin.end()}, {pads_end.begin(), pads_end.end()})
            .padval(edsl::Constant(0.0));

    // Calculate scales
    auto input_shape = I.compute_shape().sizes();  // input_shape is the shape of data after padding
    std::vector<float> scales(input_shape.size(), 1.0);
    for (auto axis : axes) {
      switch (shape_calculation_mode) {
        case ngraph::op::v4::Interpolate::ShapeCalcMode::sizes:
          scales[axis] = 1.0 * result_shape[axis] / input_shape[axis];
          break;
        case ngraph::op::v4::Interpolate::ShapeCalcMode::scales:
          scales[axis] = default_scales[axis];
          break;
        default:
          THROW_IE_EXCEPTION << "Unsupported Interpolate ShapeCalcMode";
          break;
      }
    }

    // Get output by iteratively gathering axes
    for (auto axis : axes) {
      auto IX = get_coordinate_transformed_indices(input_shape[axis], scales[axis], coordinate_transformation_mode);
      bool is_downsample = scales[axis] < 1 ? true : false;
      I = edsl::gather(I, IX)
              .axis(axis)
              .interpolationMode(get_plaidml_interpolation_mode(mode))
              .nearestMode(get_plaidml_nearest_mode(nearest_mode, is_downsample))
              .cubeCoeff(cube_coeff);
    }
    return edsl::make_tuple(I);
  });
}

}  // namespace PlaidMLPlugin
