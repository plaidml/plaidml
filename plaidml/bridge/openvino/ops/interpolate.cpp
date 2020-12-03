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

edsl::Tensor get_output_coordinate_transformed_indices(
    edsl::TensorDim I_dim, int64_t O_dim_size,
    ngraph::op::v4::Interpolate::CoordinateTransformMode coordinate_transformation_mode) {
  edsl::TensorDim O_dim(O_dim_size);
  auto IX = edsl::cast(edsl::index({O_dim}, 0), DType::FLOAT32);
  switch (coordinate_transformation_mode) {
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn:
      IX = (IX + 0.5) * I_dim / O_dim;
      break;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel:
      if (O_dim_size > 1) {
        IX = (IX + 0.5) * I_dim / O_dim - 0.5;
      }
      break;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel:
      IX = (IX + 0.5) * I_dim / O_dim - 0.5;
      break;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric:
      IX = IX * I_dim / O_dim;
      break;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners:
      if (O_dim_size > 1) {
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
    auto result_shape = cast_constant_operand<int64_t>(1, layer);
    auto scales = cast_constant_operand<float>(2, layer);
    auto mode = layer->get_attrs().mode;
    auto nearest_mode = layer->get_attrs().nearest_mode;
    auto cube_coeff = layer->get_attrs().cube_coeff;
    auto coordinate_transformation_mode = layer->get_attrs().coordinate_transformation_mode;

    bool is_downsample = false;
    for (auto scale : scales) {
      is_downsample = is_downsample || (scale < 1.0);
    }

    auto I = ctx.operands.at(0);
    auto pads_begin = layer->get_attrs().pads_begin;
    auto pads_end = layer->get_attrs().pads_end;
    I = op::explicit_padding(I, {pads_begin.begin(), pads_begin.end()}, {pads_end.begin(), pads_end.end()})
            .padval(edsl::Constant(0.0));

    std::vector<edsl::TensorDim> I_dims(I.rank());
    I.bind_dims(I_dims);

    for (int i = 0; i < result_shape.size(); i++) {
      auto IX = get_output_coordinate_transformed_indices(I_dims[i], result_shape[i], coordinate_transformation_mode);
      I = edsl::gather(I, IX)
              .axis(i)
              .interpolationMode(get_plaidml_interpolation_mode(mode))
              .nearestMode(get_plaidml_nearest_mode(nearest_mode, is_downsample))
              .cubeCoeff(cube_coeff);
    }
    return edsl::make_tuple(I);
  });
}

}  // namespace PlaidMLPlugin
