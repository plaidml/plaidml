// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <iostream>

#include "ngraph/opsets/opset3.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerROIAlign() {
  registerOp("ROIAlign", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 3);
    auto X = ctx.operands.at(0);
    auto* layer = ngraph::as_type<ngraph::opset3::ROIAlign>(ctx.layer);
    auto boxes = cast_constant_operand<int32_t>(1, layer);
    auto batch_indices = cast_constant_operand<int32_t>(2, layer);

    auto sampling_ratio = layer->get_sampling_ratio();
    IE_ASSERT(sampling_ratio == 0);

    auto pooled_h = layer->get_pooled_h();
    auto pooled_w = layer->get_pooled_w();
    auto spatial_scale = layer->get_spatial_scale();
    auto mode = layer->get_mode();
    auto num_rois = static_cast<int32_t>(batch_indices.size());

    op::PoolMode pool_mode = op::PoolMode::AVG;
    switch (mode) {
      case ngraph::opset3::ROIAlign::PoolingMode::AVG:
        pool_mode = op::PoolMode::AVG;
        break;
      case ngraph::opset3::ROIAlign::PoolingMode::MAX:
        pool_mode = op::PoolMode::MAX;
        break;
      default:
        throw std::runtime_error("Unsupported Pooling Mode");
    }

    std::vector<edsl::Tensor> pooled_rois;
    for (int i = 0; i < num_rois; i++) {
      auto x_1 = spatial_scale * boxes[4 * i];
      auto y_1 = spatial_scale * boxes[4 * i + 1];
      auto x_2 = spatial_scale * boxes[4 * i + 2];
      auto y_2 = spatial_scale * boxes[4 * i + 3];
      auto roi_width = std::max((x_2 - x_1), 1.0f);
      auto roi_height = std::max((y_2 - y_1), 1.0f);

      auto total_sampling_h = 2 * pooled_w;
      auto total_sampling_w = 2 * pooled_h;
      auto interval_h = roi_height / total_sampling_h;
      auto interval_w = roi_width / total_sampling_w;
      auto indices_h = edsl::index({edsl::TensorDim(total_sampling_h)}, 0) * interval_h + x_1 + interval_h / 2;
      auto indices_w = edsl::index({edsl::TensorDim(total_sampling_w)}, 0) * interval_w + y_1 + interval_w / 2;

      auto batch_idx = edsl::index({edsl::TensorDim(1)}, 0) + i;
      auto batch_X = edsl::gather(X, batch_idx).axis(0).interpolationMode(edsl::InterpolationMode::NEAREST);

      auto gather_w = edsl::gather(batch_X, indices_w).axis(3).interpolationMode(edsl::InterpolationMode::LINEAR);
      auto gather_h = edsl::gather(gather_w, indices_h).axis(2).interpolationMode(edsl::InterpolationMode::LINEAR);
      auto pooled_T =
          op::pool(gather_h, pool_mode, {2, 2}, {2, 2}, op::AutoPadMode::VALID, {}, op::TensorLayout::NCX, true, true);
      pooled_rois.push_back(pooled_T);
    }

    auto result = op::concatenate(pooled_rois, 0);
    return edsl::make_tuple(result);
  });
}

}  // namespace PlaidMLPlugin
