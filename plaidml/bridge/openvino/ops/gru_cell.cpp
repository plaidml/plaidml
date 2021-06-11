// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset4.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"
#include <limits>

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerGruCell() {
  registerOp("GruCell", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 5);
    auto Xt = ctx.operands.at(0);    // input tensor
    auto Ht_1 = ctx.operands.at(1);  // hidden state tensor
    auto W = ctx.operands.at(2);     // weight tensor [3*hidden_size, input_size]
    auto R = ctx.operands.at(3);     // recurrence weight tensor [3*hidden_size, input_size]
    auto B = ctx.operands.at(4);     // bias tensor linear_before_reset ? [4*hidden_size] : [3*hidden_size]

    auto input_size = Xt.compute_shape().sizes().back();
    auto* layer = ngraph::as_type<ngraph::opset4::GRUCell>(ctx.layer);
    auto hidden_size = layer->get_hidden_size();

    auto activations = layer->get_activations();
    auto activation_f = activations.at(0);
    auto activation_g = activations.at(1);

    // TODO: activation_alpha and activation_beta are not used
    auto activations_alpha = layer->get_activations_alpha();
    auto activations_beta = layer->get_activations_beta();

    auto clip = layer->get_clip();
    auto should_clip = (clip > 0.f) && (clip != std::numeric_limits<float>::infinity());

    auto linear_before_reset = layer->get_linear_before_reset();

    auto Wz = op::slice(W).add_dim(0, hidden_size).add_dim(0, input_size);
    auto Rz = op::slice(R).add_dim(0, hidden_size).add_dim(0, hidden_size);
    auto Bz = op::slice(B).add_dim(0, hidden_size);
    auto Tz = op::dot(Xt, op::transpose(Wz)) + op::dot(Ht_1, op::transpose(Rz)) + Bz;
    auto zt = clip_activation(activation_f, should_clip, clip, Tz);

    auto Wr = op::slice(W).add_dim(hidden_size, 2 * hidden_size).add_dim(0, input_size);
    auto Rr = op::slice(R).add_dim(hidden_size, 2 * hidden_size).add_dim(0, hidden_size);
    auto Br = op::slice(B).add_dim(hidden_size, 2 * hidden_size);
    auto Tr = op::dot(Xt, op::transpose(Wr)) + op::dot(Ht_1, op::transpose(Rr)) + Br;
    auto rt = clip_activation(activation_f, should_clip, clip, Tr);

    auto Wh = op::slice(W).add_dim(2 * hidden_size, 3 * hidden_size).add_dim(0, input_size);
    auto Rh = op::slice(R).add_dim(2 * hidden_size, 3 * hidden_size).add_dim(0, hidden_size);
    edsl::Tensor Th;
    if (linear_before_reset) {
      auto Bhw = op::slice(B).add_dim(2 * hidden_size, 3 * hidden_size);
      auto Bhr = op::slice(B).add_dim(3 * hidden_size, 4 * hidden_size);
      Th = op::dot(Xt, op::transpose(Wh)) + rt * (op::dot(Ht_1, op::transpose(Rh)) + Bhr) + Bhw;
    } else {
      auto Bh = op::slice(B).add_dim(2 * hidden_size, 3 * hidden_size);
      Th = op::dot(Xt, op::transpose(Wh)) + op::dot(rt * Ht_1, op::transpose(Rh)) + Bh;
    }
    auto ht_tilde = clip_activation(activation_g, should_clip, clip, Th);
    auto Ht = (1 - zt) * ht_tilde + zt * Ht_1;

    return edsl::make_tuple(Ht);
  });
}

}  // namespace PlaidMLPlugin
