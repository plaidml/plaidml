// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "ngraph/opsets/opset4.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("rnncell", [](const Context& ctx) {
  IE_ASSERT(ctx.operands.size() == 5);
  auto Xt = ctx.operands.at(0);    // input tensor
  auto Ht_1 = ctx.operands.at(1);  // hidden state tensor
  auto Wi = ctx.operands.at(2);    // weight tensor [hidden_size, input_size]
  auto Ri = ctx.operands.at(3);    // recurrence weight tensor [hidden_size, input_size]
  auto Bi = ctx.operands.at(4);    // bias tensor [hidden_size]

  auto input_size = Xt.compute_shape().sizes().back();
  auto* layer = ngraph::as_type<ngraph::opset4::RNNCell>(ctx.layer);
  auto hidden_size = layer->get_hidden_size();

  auto activations = layer->get_activations();
  auto activation = activations.at(0);

  // TODO: activation_alpha and activation_beta are not used
  auto activations_alpha = layer->get_activations_alpha();
  auto activations_beta = layer->get_activations_beta();

  auto clip = layer->get_clip();
  auto should_clip = (clip > 0.f) && (clip != std::numeric_limits<float>::infinity());

  auto Ti = op::dot(Xt, op::transpose(Wi)) + op::dot(Ht_1, op::transpose(Ri)) + Bi;
  auto Ht = clip_activation(activation, should_clip, clip, Ti);

  return edsl::make_tuple(Ht);
});

}  // namespace PlaidMLPlugin
