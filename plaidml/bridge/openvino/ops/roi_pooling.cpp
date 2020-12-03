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
    THROW_IE_EXCEPTION << "Dynamic slicing not currently supported by PlaidML plugin; all of begin, end, and stride "
                          "must be Constants.";
  }
}

// TODO need to return new height and width
edsl::Tensor crop_resized(edsl::Tensor& I, std::vector<float>& coord, float ratio, std::string method) {
  // TODO bilinear interpole ROI.
  // 5 element tuples: [batch_id, x_1, y_1, x_2, y_2]
  auto batch_id = static_cast<int>(coord[0]);
  int w_start = 0, h_start = 0, w_end = 0, h_end = 0;

  h_start = static_cast<int>(std::floor(coord[1]));
  w_start = static_cast<int>(coord[2]);
  h_end = static_cast<int>(coord[3]);
  w_end = static_cast<int>(coord[4]);

  auto channel = I.compute_shape().sizes()[1];
  edsl::Tensor corp_tensor = op::slice(I)
                                 .add_dim(batch_id)         //
                                 .add_dim(0, channel)       //
                                 .add_dim(h_start, h_end)   //
                                 .add_dim(w_start, w_end);  //
  return op::unsqueeze(corp_tensor, {0});
}

}  // namespace

namespace PlaidMLPlugin {

void registerROIPooling() {
  registerOp("ROIPooling", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::ROIPooling>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);

    auto pooled_shape = layer->get_output_size();
    auto pooled_height = pooled_shape[0];
    auto pooled_width = pooled_shape[1];
    auto spatial_ratio = layer->get_spatial_scale();
    auto method = layer->get_method();

    const static int BOX_ELEMENT_SIZE = 5;
    auto coords_box = cast_constant_operand<float>(1, layer);
    IE_ASSERT((coords_box.size() % BOX_ELEMENT_SIZE) == 0);

    std::vector<edsl::Tensor> ROI_pools;

    // 2D input tensor of shape [NUM_ROIS, 5] describing box
    // consisting of 5 element tuples: [batch_id, x_1, y_1, x_2, y_2]
    for (auto index = coords_box.begin(); index != coords_box.end(); index += BOX_ELEMENT_SIZE) {
      std::vector<float> coord(index, index + BOX_ELEMENT_SIZE);
      if (method == "BILINEAR") {
        for (int i = 1; i < coord.size(); i++) {
          coord[i] *= spatial_ratio;
        }
      }
      auto ROI_tensor = crop_resized(I, coord, spatial_ratio, method);
      // TODO get resized height and width. define kernel size.
      auto kernel_height = static_cast<int>((coord[3] - coord[1]) / pooled_height);
      auto kernel_width = static_cast<int>((coord[4] - coord[2]) / pooled_width);
      auto pooled_tensor = op::pool(ROI_tensor,                                //
                                    op::PoolMode::MAX,               //
                                    {kernel_height, kernel_width},    //
                                    {kernel_height, kernel_width},      //
                                    op::AutoPadMode::VALID,       //
                                    {},                                        //
                                    op::TensorLayout::NCX);

      ROI_pools.push_back(pooled_tensor);
    }

    auto O = op::concatenate(ROI_pools, 0);
    return edsl::make_tuple(O);
  });
}
}  // namespace PlaidMLPlugin
