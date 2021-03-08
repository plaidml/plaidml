// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset3.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

namespace {

op::PoolMode to_plaidml(ngraph::op::v3::ROIAlign::PoolingMode mode) {
  switch (mode) {
    case ngraph::opset3::ROIAlign::PoolingMode::AVG: {
      return op::PoolMode::AVG;
    }
    case ngraph::opset3::ROIAlign::PoolingMode::MAX: {
      return op::PoolMode::MAX;
    }
    default:
      // TODO: Verify these are the unsupported types
      THROW_IE_EXCEPTION << "Unsupported ROIAlign pooling mode";
  }
}

edsl::Tensor get_roi(edsl::Tensor batch, std::vector<float>& coords, int roi_index, int pooled_h, int pooled_w,
                     int sampling_ratio, float spatial_scale, op::PoolMode pool_mode) {
  float w1 = round(spatial_scale * coords[4 * roi_index]);
  float h1 = round(spatial_scale * coords[4 * roi_index + 1]);
  float w2 = round(spatial_scale * coords[4 * roi_index + 2]);
  float h2 = round(spatial_scale * coords[4 * roi_index + 3]);

  auto roi_height = std::max((h2 - h1), 1.0f);
  auto roi_width = std::max((w2 - w1), 1.0f);
  auto bin_height = roi_height / pooled_h;
  auto bin_width = roi_width / pooled_w;

  auto sampling_ratio_h = sampling_ratio == 0 ? static_cast<int>(ceil(bin_height)) : sampling_ratio;
  auto sampling_ratio_w = sampling_ratio == 0 ? static_cast<int>(ceil(bin_width)) : sampling_ratio;
  auto sample_distance_h = bin_height / static_cast<float>(sampling_ratio_h);
  auto sample_distance_w = bin_width / static_cast<float>(sampling_ratio_w);

  auto IX_h = edsl::cast(edsl::index({edsl::TensorDim(sampling_ratio_h * pooled_h)}, 0), DType::FLOAT32);
  IX_h = h1 + (IX_h + 0.5) * sample_distance_h;
  auto IX_w = edsl::cast(edsl::index({edsl::TensorDim(sampling_ratio_w * pooled_w)}, 0), DType::FLOAT32);
  IX_w = w1 + (IX_w + 0.5) * sample_distance_w;

  edsl::Tensor roi;
  switch (pool_mode) {
    case op::PoolMode::AVG: {
      auto batch_gathered_w = edsl::gather(batch, IX_h).axis(2).interpolationMode(edsl::InterpolationMode::LINEAR);
      roi = edsl::gather(batch_gathered_w, IX_w).axis(3).interpolationMode(edsl::InterpolationMode::LINEAR);
      break;
    }
    case op::PoolMode::MAX: {
      std::vector<std::pair<edsl::Tensor, edsl::Tensor>> IX_hs;
      auto IX_h_floor = edsl::floor(IX_h);
      auto IX_h_ceil = IX_h_floor + 1;
      IX_hs.push_back({IX_h_floor, IX_h_ceil - IX_h});
      IX_hs.push_back({IX_h_ceil, IX_h - IX_h_floor});

      std::vector<std::pair<edsl::Tensor, edsl::Tensor>> IX_ws;
      auto IX_w_floor = edsl::floor(IX_w);
      auto IX_w_ceil = IX_w_floor + 1;
      IX_ws.push_back({IX_w_floor, IX_w_ceil - IX_w});
      IX_ws.push_back({IX_w_ceil, IX_w - IX_w_floor});

      std::vector<edsl::Tensor> roi_corners;
      for (auto ih : IX_hs) {
        for (auto iw : IX_ws) {
          auto batch_gathered_w = edsl::gather(batch, ih.first).axis(2);
          edsl::Tensor roi_corner = edsl::gather(batch_gathered_w, iw.first).axis(3);

          auto shape = roi_corner.compute_shape().sizes();
          std::vector<int> shape_int(shape.begin(), shape.end());
          roi_corner = roi_corner * op::broadcast(ih.second, shape_int, {2}) * op::broadcast(iw.second, shape_int, {3});

          shape_int.push_back(1);
          roi_corner = op::reshape(roi_corner, edsl::make_tuple(shape_int));
          roi_corners.push_back(roi_corner);
        }
      }
      roi = op::max(op::concatenate(roi_corners, 4), edsl::Value(4));
      break;
    }
    default:
      throw std::runtime_error("Unsupported ROIAlign pooling mode");
  }
  return op::pool(roi, pool_mode, {sampling_ratio_h, sampling_ratio_w}, {sampling_ratio_h, sampling_ratio_w},
                  op::AutoPadMode::VALID, {}, op::TensorLayout::NCX);
}

}  // namespace

void registerROIAlign() {
  registerOp("ROIAlign", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 3);
    auto I = ctx.operands.at(0);
    auto* layer = ngraph::as_type<ngraph::opset3::ROIAlign>(ctx.layer);
    auto coords = cast_constant_operand<float>(1, layer);
    auto batch_indices = cast_constant_operand<int32_t>(2, layer);
    auto num_rois = static_cast<int32_t>(batch_indices.size());

    auto pooled_h = layer->get_pooled_h();
    auto pooled_w = layer->get_pooled_w();
    auto sampling_ratio = layer->get_sampling_ratio();
    auto spatial_scale = layer->get_spatial_scale();
    auto pool_mode = to_plaidml(layer->get_mode());

    std::vector<edsl::Tensor> rois;
    for (int i = 0; i < num_rois; i++) {
      auto batch_idx = edsl::index({edsl::TensorDim(1)}, 0) + batch_indices[i];
      auto batch = edsl::gather(I, batch_idx).axis(0);
      edsl::Tensor roi = get_roi(batch, coords, i, pooled_h, pooled_w, sampling_ratio, spatial_scale, pool_mode);
      rois.push_back(roi);
    }

    return edsl::make_tuple(op::concatenate(rois, 0));
  });
}

}  // namespace PlaidMLPlugin
