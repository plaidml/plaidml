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

edsl::Tensor crop_max_pooling(edsl::Tensor I, const std::vector<edsl::Tensor>& coord, int64_t pooled_h,
                              int64_t pooled_w, int height, int width) {
  auto roi_w_start = coord[0];
  auto roi_h_start = coord[1];
  auto roi_w_end = coord[2];
  auto roi_h_end = coord[3];

  auto roi_width = op::maximum(roi_w_end - roi_w_start + 1, edsl::cast(edsl::Tensor(1.0), roi_w_start.dtype()));
  auto roi_height = op::maximum(roi_h_end - roi_h_start + 1, edsl::cast(edsl::Tensor(1.0), roi_h_start.dtype()));

  auto bin_size_h = roi_height / pooled_h;
  auto bin_size_w = roi_width / pooled_w;

  auto shapes = I.compute_shape().sizes();
  std::vector<edsl::Tensor> pooled_tensor;
  for (auto i = 0; i < pooled_h; i++) {
    for (auto j = 0; j < pooled_w; j++) {
      auto h1 = roi_h_start + edsl::floor(bin_size_h * i);
      auto w1 = roi_w_start + edsl::floor(bin_size_w * j);
      auto h2 = roi_h_start + edsl::ceil(bin_size_h * (i + 1));
      auto w2 = roi_w_start + edsl::ceil(bin_size_w * (j + 1));

      auto start_h = op::minimum(op::maximum(h1, edsl::cast(edsl::Tensor(0), h1.dtype())),
                                 edsl::cast(edsl::Tensor(height), h1.dtype()));
      auto start_w = op::minimum(op::maximum(w1, edsl::cast(edsl::Tensor(0), w1.dtype())),
                                 edsl::cast(edsl::Tensor(width), w1.dtype()));
      auto end_h = op::minimum(op::maximum(h2, edsl::cast(edsl::Tensor(0), h2.dtype())),
                               edsl::cast(edsl::Tensor(height), h2.dtype()));
      auto end_w = op::minimum(op::maximum(w2, edsl::cast(edsl::Tensor(0), w2.dtype())),
                               edsl::cast(edsl::Tensor(width), w2.dtype()));

      // if start equal to end, we have to guarantee that index is at least one. and they are not over the border.
      auto delta_w = end_w - start_w;
      auto slice_w_index = edsl::select(delta_w > 0, delta_w, edsl::cast(edsl::Tensor(1), delta_w.dtype()));
      auto delta_h = end_h - start_h;
      auto slice_h_index = edsl::select(delta_h > 0, delta_h, edsl::cast(edsl::Tensor(1), delta_h.dtype()));
      auto h_index = edsl::select(start_h >= height, edsl::cast(edsl::Tensor(height - 1), start_h.dtype()), start_h);
      auto w_index = edsl::select(start_w >= width, edsl::cast(edsl::Tensor(width - 1), start_w.dtype()), start_w);

      // h_tensor/w_tensor has 1D dynamic shape of size slice_h_index/slice_w_index.
      // Here is a workaround to use a 1D tensor of size height/weight and fill the redundant memory in the tensor with
      // the first valid element. The result is the max element of the Tensor, so the result is still correct.
      auto h_tensor = edsl::cast(edsl::index({edsl::TensorDim(height)}, 0), h_index.dtype());
      auto w_tensor = edsl::cast(edsl::index({edsl::TensorDim(width)}, 0), w_index.dtype());
      h_tensor = edsl::select(h_tensor >= h_index && h_tensor < h_index + slice_h_index, h_tensor, h_index);
      w_tensor = edsl::select(w_tensor >= w_index && w_tensor < w_index + slice_w_index, w_tensor, w_index);
      auto gather_w = edsl::gather(I, w_tensor).axis(3);
      edsl::Tensor crop = edsl::gather(gather_w, h_tensor).axis(2);

      // Get max value from bin, then put it to pooled tensor.
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

edsl::Tensor bilinear_pooling(edsl::Tensor I, const std::vector<edsl::Tensor>& coord, int64_t pooled_h,
                              int64_t pooled_w, int height, int width) {
  auto roi_w_start = coord[0];
  auto roi_h_start = coord[1];
  auto roi_w_end = coord[2];
  auto roi_h_end = coord[3];

  auto roi_width = (roi_w_end - roi_w_start) * (width - 1);
  auto roi_height = (roi_h_end - roi_h_start) * (height - 1);
  auto roi_h_scale = roi_height / (pooled_h - 1);
  auto roi_w_scale = roi_width / (pooled_w - 1);

  // get center point of every ROI bin.
  auto in_h = edsl::cast(edsl::index({edsl::TensorDim(pooled_h)}, 0), DType::FLOAT32) * roi_h_scale +
              roi_h_start * (height - 1);
  auto in_w =
      edsl::cast(edsl::index({edsl::TensorDim(pooled_w)}, 0), DType::FLOAT32) * roi_w_scale + roi_w_start * (width - 1);
  auto I_gathered_h = edsl::gather(I, in_h).interpolationMode(edsl::InterpolationMode::LINEAR).axis(2);
  return edsl::gather(I_gathered_h, in_w).interpolationMode(edsl::InterpolationMode::LINEAR).axis(3);
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

    auto coords_box = ctx.operands.at(1);
    auto box_shape = coords_box.compute_shape().sizes();

    auto shapes = I.compute_shape().sizes();
    auto height = shapes[2];
    auto width = shapes[3];

    std::vector<edsl::Tensor> ROI_pools;
    edsl::Tensor ZERO = edsl::index({edsl::TensorDim(1)}, 0);
    // 2D input tensor of shape [NUM_ROIS, 5] describing box
    for (int i = 0; i < box_shape[0]; i++) {
      edsl::Tensor box = edsl::gather(coords_box, i);
      box = edsl::reshape(box, std::vector<int64_t>{BOX_ELEMENT_SIZE});
      auto batch_id = edsl::gather(box, 0);
      auto slice_I = edsl::gather(I, batch_id).axis(0);
      std::vector<edsl::Tensor> coord;
      for (int j = 1; j < BOX_ELEMENT_SIZE; j++) {
        coord.push_back(edsl::gather(box, j));
      }

      edsl::Tensor pooled_tensor;
      if (method == "max") {
        // translate ROI coordinates from their input normalize scale to feature map scale.
        for (int i = 0; i < coord.size(); i++) {
          coord[i] = edsl::round(coord[i] * spatial_ratio);
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
