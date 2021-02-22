// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace {

edsl::Tensor crop_max_pooling(edsl::Tensor I, const std::vector<float>& coord, int64_t pooled_h, int64_t pooled_w,
                              int height, int width) {
  auto roi_w_start = coord[0];
  auto roi_h_start = coord[1];
  auto roi_w_end = coord[2];
  auto roi_h_end = coord[3];

  float roi_width = std::max(roi_w_end - roi_w_start + 1, 1.0f);
  float roi_height = std::max(roi_h_end - roi_h_start + 1, 1.0f);

  float bin_size_h = roi_height / pooled_h;
  float bin_size_w = roi_width / pooled_w;

  auto shapes = I.compute_shape().sizes();
  std::vector<edsl::Tensor> pooled_tensor;
  for (auto i = 0; i < pooled_h; i++) {
    for (auto j = 0; j < pooled_w; j++) {
      // enlarge bin.
      int h1 = roi_h_start + std::floor(bin_size_h * i);
      int w1 = roi_w_start + std::floor(bin_size_w * j);
      int h2 = roi_h_start + std::ceil(bin_size_h * (i + 1));
      int w2 = roi_w_start + std::ceil(bin_size_w * (j + 1));

      // check border.
      auto start_h = std::min(std::max(h1, 0), height);
      auto start_w = std::min(std::max(w1, 0), width);
      auto end_h = std::min(std::max(h2, 0), height);
      auto end_w = std::min(std::max(w2, 0), width);

      // if start equal to end, we have to guarantee that index is at least one. and they are not over the border.
      auto slice_w_index = end_w - start_w > 0 ? end_w - start_w : 1;
      auto slice_h_index = end_h - start_h > 0 ? end_h - start_h : 1;
      auto h_index = start_h >= height ? height - 1 : start_h;
      auto w_index = start_w >= width ? width - 1 : start_w;
      auto h_tensor = edsl::index({edsl::TensorDim(slice_h_index)}, 0) + h_index;
      auto w_tensor = edsl::index({edsl::TensorDim(slice_w_index)}, 0) + w_index;
      auto gather_w = edsl::gather(I, w_tensor).axis(3);
      edsl::Tensor crop = edsl::gather(gather_w, h_tensor).axis(2);

      // get max value from bin, then put it to pooled tensor.
      std::vector<edsl::TensorDim> dims(crop.rank());
      crop.bind_dims(dims);
      std::vector<edsl::TensorIndex> idx(crop.rank());
      edsl::Tensor bin_max =
          edsl::Contraction().outShape({dims[0], dims[1]}).outAccess({idx[0], idx[1]}).max(crop(idx));
      pooled_tensor.push_back(edsl::reshape(bin_max, {shapes[0], shapes[1], 1}));
    }
  }

  return edsl::reshape(op::concatenate(pooled_tensor, 2), {shapes[0], shapes[1], pooled_h, pooled_w});
}

edsl::Tensor bilinear_pooling(edsl::Tensor I, const std::vector<float>& coord, int64_t pooled_h, int64_t pooled_w,
                              int height, int width) {
  auto roi_w_start = coord[0];
  auto roi_h_start = coord[1];
  auto roi_w_end = coord[2];
  auto roi_h_end = coord[3];

  float roi_width = (roi_w_end - roi_w_start) * (width - 1);
  float roi_height = (roi_h_end - roi_h_start) * (height - 1);

  float roi_h_scale = roi_height / (pooled_h - 1);
  float roi_w_scale = roi_width / (pooled_w - 1);

  // get center point of every ROI bin.
  auto in_h = edsl::cast(edsl::index({edsl::TensorDim(pooled_h)}, 0), DType::FLOAT32) * roi_h_scale +
              roi_h_start * (height - 1);
  auto in_w =
      edsl::cast(edsl::index({edsl::TensorDim(pooled_w)}, 0), DType::FLOAT32) * roi_w_scale + roi_w_start * (width - 1);

  auto I_gathered_h = edsl::gather(I, in_h).axis(2);
  return edsl::gather(I_gathered_h, in_w).axis(3);
}

}  // namespace

namespace PlaidMLPlugin {

const static int BOX_ELEMENT_SIZE = 5;  // NOLINT

void registerROIPooling() {
  registerOp("ROIPooling", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::ROIPooling>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);

    auto pooled_shape = layer->get_output_size();
    auto pooled_height = static_cast<int64_t>(pooled_shape[0]);
    auto pooled_width = static_cast<int64_t>(pooled_shape[1]);
    auto spatial_ratio = layer->get_spatial_scale();
    auto method = layer->get_method();

    auto coords_box = cast_constant_operand<float>(1, layer);
    IE_ASSERT((coords_box.size() % BOX_ELEMENT_SIZE) == 0);

    auto shapes = I.compute_shape().sizes();
    auto height = shapes[2];
    auto width = shapes[3];

    std::vector<edsl::Tensor> ROI_pools;
    // 2D input tensor of shape [NUM_ROIS, 5] describing box
    for (auto index = coords_box.begin(); index != coords_box.end(); index += BOX_ELEMENT_SIZE) {
      // consisting of 5 element tuples: [batch_id, x_1, y_1, x_2, y_2]
      auto batch_id = *index;
      std::vector<float> coord(index + 1, index + BOX_ELEMENT_SIZE);

      auto batch_indices = edsl::index({edsl::TensorDim(1)}, 0) + batch_id;
      auto slice_I = edsl::gather(I, batch_indices).axis(0);

      edsl::Tensor pooled_tensor;
      if (method == "max") {
        // translate ROI coordinates from their input normalize scale to feature map scale.
        for (int i = 0; i < coord.size(); i++) {
          coord[i] = std::round(coord[i] * spatial_ratio);
        }
        pooled_tensor = crop_max_pooling(slice_I, coord, pooled_height, pooled_width, height, width);
      } else if (method == "bilinear") {
        // follow ngraph implementation, which doesn't use spatial_ratio in "bilinear" method.
        pooled_tensor = bilinear_pooling(slice_I, coord, pooled_height, pooled_width, height, width);
      } else {
        THROW_IE_EXCEPTION << "ROIPooling op currently only support 'max' and 'bilinear' method;";
      }
      ROI_pools.push_back(pooled_tensor);
    }

    auto O = op::concatenate(ROI_pools, 0);
    return edsl::make_tuple(O);
  });
}
}  // namespace PlaidMLPlugin
