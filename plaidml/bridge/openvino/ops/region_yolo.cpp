// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerRegionYolo() {
  registerOp("RegionYolo", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::RegionYolo>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 1);
    auto I = ctx.operands.at(0);

    auto input_shape = I.compute_shape().sizes();
    size_t batches = input_shape[0];
    size_t height = input_shape[2];
    size_t width = input_shape[3];

    auto coords = layer->get_num_coords();
    auto classes = layer->get_num_classes();
    auto regions = layer->get_num_regions();
    auto do_softmax = layer->get_do_softmax();
    auto mask = layer->get_mask();
    auto mask_size = mask.size();

    size_t num_regions = 0;
    size_t end_index = 0;

    if (do_softmax) {
      // Region layer (Yolo v2)
      num_regions = regions;
      end_index = 1;
    } else {
      // Yolo layer (Yolo v3)
      num_regions = mask_size;
      end_index = classes + 1;
    }

    std::vector<int64_t> yolo_shape = {static_cast<int64_t>(batches), static_cast<int64_t>(num_regions),
                                       static_cast<int64_t>(classes + coords + 1), static_cast<int64_t>(height),
                                       static_cast<int64_t>(width)};
    edsl::Tensor O = edsl::reshape(I, yolo_shape);

    int64_t index_axis = 2;

    auto IX = edsl::cast(edsl::index({edsl::TensorDim(2)}, 0), DType::INT32);
    edsl::Tensor O_update = op::sigmoid(edsl::gather(O, IX).axis(index_axis));
    O = edsl::scatter(O, IX, O_update).axis(index_axis).mode(edsl::ScatterMode::UPDATE_SLICE);

    IX = edsl::cast(edsl::index({edsl::TensorDim(end_index)}, 0), DType::INT32) + coords;
    O_update = op::sigmoid(edsl::gather(O, IX).axis(index_axis));
    O = edsl::scatter(O, IX, O_update).axis(index_axis).mode(edsl::ScatterMode::UPDATE_SLICE);

    if (do_softmax) {
      edsl::TensorDim O_dim(classes);
      IX = edsl::cast(edsl::index({O_dim}, 0), DType::INT32) + coords + 1;
      O_update = op::softmax(edsl::gather(O, IX).axis(index_axis), index_axis);
      O = edsl::scatter(O, IX, O_update).axis(index_axis).mode(edsl::ScatterMode::UPDATE_SLICE);
    }

    return edsl::make_tuple(edsl::reshape(O, input_shape));
  });
}
}  // namespace PlaidMLPlugin
