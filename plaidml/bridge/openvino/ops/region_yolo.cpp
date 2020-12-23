// Copyright (C) 2020 Intel Corporation
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
    const int batches = input_shape[0];
    const int channels = input_shape[1];
    const int height = input_shape[2];
    const int width = input_shape[3];

    auto coords = layer->get_num_coords();
    auto classes = layer->get_num_classes();
    auto regions = layer->get_num_regions();
    auto do_softmax = layer->get_do_softmax();
    auto mask = layer->get_mask();
    auto mask_size = mask.size();

    edsl::Tensor O = I;

    int num_regions = 0;
    int end_index = 0;

    if (do_softmax) {
      // Region layer (Yolo v2)
      num_regions = regions;
      end_index = width * height;
    } else {
      // Yolo layer (Yolo v3)
      num_regions = mask_size;
      end_index = width * height * (classes + 1);
    }

    const int inputs_size = width * height * num_regions * (classes + coords + 1);

    return edsl::make_tuple(O);
  });
}

}  // namespace PlaidMLPlugin
