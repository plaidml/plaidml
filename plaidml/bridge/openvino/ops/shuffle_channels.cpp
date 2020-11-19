// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("shuffleChannels", [](const Context& ctx) {
  auto* layer = ngraph::as_type<ngraph::opset3::ShuffleChannels>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  auto group = layer->get_group();
  auto axis = layer->get_axis();
  if (axis < 0) {
    axis = axis + 4;
  }

  std::vector<edsl::TensorDim> original_dims(I.rank());
  I.bind_dims(original_dims);

  std::vector<edsl::TensorDim> channel_group_dims(original_dims);
  channel_group_dims[axis] = original_dims[axis] / group;
  channel_group_dims.emplace(channel_group_dims.begin() + axis, group);
  auto reshape_I = edsl::reshape(I, channel_group_dims);

  std::vector<edsl::Value> order(channel_group_dims.size());
  for (size_t i = 0; i < order.size(); i++) {
    order[i] = edsl::Value(i);
  }
  std::swap(order[axis], order[axis + 1]);
  auto transpose_I = op::transpose(reshape_I, edsl::Value(order));

  auto O = edsl::reshape(transpose_I, original_dims);
  return edsl::make_tuple(O);
});

}  // namespace PlaidMLPlugin
