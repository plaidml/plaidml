// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset3.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include <algorithm>
#include <cmath>

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
template <typename T>
edsl::Tensor make_tensor(DType dtype, const std::vector<int64_t>& dims, const std::vector<T>& data,
                         const std::string& name) {
  TensorShape shape(dtype, dims);
  Buffer buffer(shape);
  buffer.copy_from(data.data());
  return edsl::Constant(buffer, name);
}
}  // namespace

namespace PlaidMLPlugin {

// TODO: change register method as plaidml-v1
static OpRegistration reg("ROIAlign", [](const Context& ctx) {
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

  // Crop, Resize and Pool
  std::vector<edsl::Tensor> pooled_rois;
  for (int i = 0; i < num_rois; i++) {
    auto x_1 = spatial_scale * boxes[4 * i];
    auto y_1 = spatial_scale * boxes[4 * i + 1];
    auto x_2 = spatial_scale * boxes[4 * i + 2];
    auto y_2 = spatial_scale * boxes[4 * i + 3];
    auto roi_width = std::max(spatial_scale * (x_2 - x_1), 1.0f);
    auto roi_height = std::max(spatial_scale * (y_2 - y_1), 1.0f);

    int pool_size_h, pool_size_w;
    if (sampling_ratio == 0) {
      pool_size_h = std::ceil(roi_height / pooled_h);
      pool_size_w = std::ceil(roi_width / pooled_w);
    } else {
      pool_size_h = pool_size_w = sampling_ratio;
    }
    auto sampling_h = pool_size_h * pooled_h;
    auto sampling_w = pool_size_w * pooled_w;

    auto interval_h = roi_width / (sampling_h + 1.0);
    auto interval_w = roi_width / (sampling_w + 1.0);

    std::vector<int32_t> indices_h(sampling_h);
    for (int ih = 0; ih < sampling_h; ih++) {
      indices_h.push_back(x_1 + interval_h * ih);
    }
    auto ind_tensor_h = make_tensor(DType::INT32, {sampling_h}, indices_h, "indices_h");
    std::vector<int64_t> indices_w(sampling_w);
    for (int iw = 0; iw < sampling_w; iw++) {
      indices_w.push_back(x_1 + interval_w * iw);
    }
    auto ind_tensor_w = make_tensor(DType::INT32, {sampling_w}, indices_w, "indices_w");

    auto slice_X = op::slice(X)
                       .add_dim(batch_indices[i])
                       .add_dim(0, input_size[1])
                       .add_dim(0, input_size[2])
                       .add_dim(0, input_size[3]);
    auto batch_X = op::unsqueeze(slice_X, {0});

    auto gather_w = edsl::gather(batch_X, ind_tensor_w).axis(3).interpolationMode(edsl::InterpolationMode::LINEAR);
    auto gather_h = edsl::gather(gather_w, ind_tensor_h).axis(2).interpolationMode(edsl::InterpolationMode::LINEAR);
    auto pooled_T = op::pool(gather_h, pool_mode, {pool_size_h, pool_size_w}, {pool_size_h, pool_size_w},
                             op::AutoPadMode::VALID, {}, op::TensorLayout::NCX, true, true);
    pooled_rois.push_back(pooled_T);
  }

  auto result = op::concatenate(pooled_rois, 0);
  return edsl::make_tuple(result);
});

}  // namespace PlaidMLPlugin