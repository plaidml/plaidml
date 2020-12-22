// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"
#include <tuple>

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
        << "Dynamic coordinate is not currently supported by PlaidML plugin; all coordinate must be Constants. ";
  }
}

edsl::Tensor crop_max_pooling(edsl::Tensor I, const std::vector<float>& coord, int64_t pooled_h, int64_t pooled_w) {
  auto x_1 = coord[0];
  auto y_1 = coord[1];
  auto x_2 = coord[2];
  auto y_2 = coord[3];

  auto roi_width = std::max(x_2 - x_1, 1.0f);
  auto roi_height = std::max(y_2 - y_1, 1.0f);

  auto bin_size_h = roi_height / static_cast<float>(pooled_h);
  auto bin_size_w = roi_width / static_cast<float>(pooled_w);

  edsl::Tensor pooled_tensor;
  for (auto i = 0; i < pooled_h; i++) {
    for (auto j = 0; j < pooled_w; j++) {
      // enlarge bin.
      auto start_h = std::floor(x_1 + bin_size_h * i);
      auto start_w = std::floor(y_1 + bin_size_w * j);
      auto end_h = std::ceil(x_1 + bin_size_h * (i + 1));
      auto end_w = std::ceil(y_1 + bin_size_w * (j + 1));

      auto h_tensor =
          edsl::index({edsl::TensorDim(static_cast<int64_t>(end_h - start_h))}, 0) + static_cast<int>(start_h);
      auto w_tensor =
          edsl::index({edsl::TensorDim(static_cast<int64_t>(end_w - start_w))}, 0) + static_cast<int>(start_w);

      // crop bin from ROI.
      auto gather_w = edsl::gather(I, w_tensor).axis(3).interpolationMode(edsl::InterpolationMode::LINEAR);
      auto crop = static_cast<edsl::Tensor>(
          edsl::gather(gather_w, h_tensor).axis(2).interpolationMode(edsl::InterpolationMode::LINEAR));

      // get max value from bin, then put it to pooled tensor.
      std::vector<edsl::TensorDim> dims(crop.rank());
      crop.bind_dims(dims);
      std::vector<edsl::TensorIndex> idx(crop.rank());
      auto bin_max = edsl::Contraction()
                         .outShape({dims[0], dims[1], edsl::TensorDim(pooled_h), edsl::TensorDim(pooled_w)})
                         .outAccess({idx[0], idx[1], edsl::TensorIndex(i), edsl::TensorIndex(j)})
                         .max(crop(idx));
      pooled_tensor = (i == 0) ? bin_max : edsl::select(pooled_tensor > bin_max, pooled_tensor, bin_max);
    }
  }
  return pooled_tensor;
}

edsl::Tensor bilinear_pooling(edsl::Tensor I, const std::vector<float>& coord, int64_t pooled_h, int64_t pooled_w) {
  auto x_1 = coord[0];
  auto y_1 = coord[1];
  auto x_2 = coord[2];
  auto y_2 = coord[3];

  auto roi_width = y_2 - y_1;
  auto roi_height = x_2 - x_1;

  auto roi_h_scale = roi_height / pooled_h;
  auto roi_w_scale = roi_width / pooled_w;

  auto h_tensor = edsl::cast(edsl::index({edsl::TensorDim(pooled_h)}, 0), DType::FLOAT32) * roi_h_scale + x_1;
  auto w_tensor = edsl::cast(edsl::index({edsl::TensorDim(pooled_w)}, 0), DType::FLOAT32) * roi_w_scale + y_1;

  auto gather_w = edsl::gather(I, w_tensor).axis(3).interpolationMode(edsl::InterpolationMode::LINEAR);
  auto gather_h = edsl::gather(gather_w, h_tensor).axis(2).interpolationMode(edsl::InterpolationMode::LINEAR);

  return gather_h;
}

}  // namespace

namespace PlaidMLPlugin {

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

    const static int BOX_ELEMENT_SIZE = 5;
    auto coords_box = cast_constant_operand<float>(1, layer);
    IE_ASSERT((coords_box.size() % BOX_ELEMENT_SIZE) == 0);

    int height = 0, width = 0;
    if (method == "bilinear") {
      auto shapes = I.compute_shape().sizes();
      height = shapes[2];
      width = shapes[3];
    }

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
        for (int i = 1; i < coord.size(); i++) {
          coord[i] = std::round(coord[i] * spatial_ratio);
        }
        pooled_tensor = crop_max_pooling(slice_I, coord, pooled_height, pooled_width);
      } else if (method == "bilinear") {
        // follow ngraph implementation, which doesn't use spatial_ratio in "bilinear" method.
        coord[1] *= (width - 1);
        coord[3] *= (width - 1);
        coord[2] *= (height - 1);
        coord[4] *= (height - 1);
        pooled_tensor = bilinear_pooling(slice_I, coord, pooled_height, pooled_width);
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
