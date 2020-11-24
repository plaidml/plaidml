// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset3.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include <algorithm>

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

edsl::Tensor crop_and_resize(edsl::Tensor image, std::vector<std::vector<float>> boxes,
                             std::vector<int64_t> box_indices, std::vector<int64_t> crop_size, std::string method) {
  edsl::Tensor O;
  // TODO: implement this function and add params check
  return O;
}

}  // namespace

namespace PlaidMLPlugin {

static OpRegistration reg("roialign", [](const Context& ctx) {
  IE_ASSERT(ctx.operands.size() == 3);
  auto X = ctx.operands.at(0);
  auto* layer = ngraph::as_type<ngraph::opset3::ROIAlign>(ctx.layer);
  auto boxes = cast_constant_operand<int32_t>(1, layer);
  auto batch_indices = cast_constant_operand<int32_t>(2, layer);

  auto pooled_h = layer->get_pooled_h();
  auto pooled_w = layer->get_pooled_w();
  auto sampling_ratio = layer->get_sampling_ratio();
  auto spatial_scale = layer->get_spatial_scale();
  auto mode = layer->get_mode();
  auto num_rois = static_cast<int32_t>(batch_indices.size());

  auto input_size = X.compute_shape().sizes();

  op::PoolMode pool_mode;
  switch (mode) {
    case ngraph::op::v3::ROIAlign::PoolingMode::AVG:
      pool_mode = op::PoolMode::AVG;
      break;
    case ngraph::op::v3::ROIAlign::PoolingMode::MAX:
      pool_mode = op::PoolMode::MAX;
      break;
    default:
      std::runtime_error("Unsupported Pooling Mode");
  }

  // Crop and Resize
  edsl::Tensor resized_T = edsl::Placeholder(X.dtype(), {num_rois, input_size[1], pooled_h, pooled_w});
  for (int i = 0; i < num_rois; i++) {
    auto roi_width = std::max(spatial_scale * (boxes[4 * i + 2] - boxes[4 * i]), 1.0f);
    auto roi_height = std::max(spatial_scale * (boxes[4 * i + 3] - boxes[4 * i + 1]), 1.0f);
    auto x_1 = spatial_scale * boxes[4 * i];
    auto y_1 = spatial_scale * boxes[4 * i + 1];
    auto x_2 = spatial_scale * boxes[4 * i + 2];
    auto y_2 = spatial_scale * boxes[4 * i + 3];
  }
}

  auto result = op::pool(resized_T, pool_mode, )

  return edsl::make_tuple();
}