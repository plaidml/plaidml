// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerLstmCell() {
  registerOp("LstmCell", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 6);
    auto Xt = ctx.operands.at(0);    // input tensor
    auto Ht_1 = ctx.operands.at(1);  // hidden state tensor
    auto Ct_1 = ctx.operands.at(2);  // cell state tensor
    auto W = ctx.operands.at(3);     // weight tensor [4 * hidden_size, input_size]
    auto R = ctx.operands.at(4);     // recurrence weight tensor [4 * hidden_size, input_size]
    auto B = ctx.operands.at(5);     // bias tensor [4 * hidden_size]

    auto input_size = Xt.compute_shape().sizes().back();
    auto* layer = ngraph::as_type<ngraph::opset4::LSTMCell>(ctx.layer);
    auto hidden_size = layer->get_hidden_size();

    auto activations = layer->get_activations();
    auto activation_f = activations.at(0);
    auto activation_g = activations.at(1);
    auto activation_h = activations.at(2);

    // TODO: activation_alpha and activation_beta are not used
    auto activation_alpha = layer->get_activations_alpha();
    auto activation_beta = layer->get_activations_beta();

    auto clip = layer->get_clip();
    auto should_clip = (clip > 0.f) && (clip != std::numeric_limits<float>::infinity());

    auto Wf = op::slice(W).add_dim(0, hidden_size).add_dim(0, input_size);
    auto Rf = op::slice(R).add_dim(0, hidden_size).add_dim(0, hidden_size);
    auto Bf = op::slice(B).add_dim(0, hidden_size);
    auto Tf = op::dot(Xt, op::transpose(Wf)) + op::dot(Ht_1, op::transpose(Rf)) + Bf;
    auto ft = clip_activation(activation_f, should_clip, clip, Tf);

    auto Wi = op::slice(W).add_dim(hidden_size, 2 * hidden_size).add_dim(0, input_size);
    auto Ri = op::slice(R).add_dim(hidden_size, 2 * hidden_size).add_dim(0, hidden_size);
    auto Bi = op::slice(B).add_dim(hidden_size, 2 * hidden_size);
    auto Ti = op::dot(Xt, op::transpose(Wi)) + op::dot(Ht_1, op::transpose(Ri)) + Bi;
    auto it = clip_activation(activation_f, should_clip, clip, Ti);

    auto Wc = op::slice(W).add_dim(2 * hidden_size, 3 * hidden_size).add_dim(0, input_size);
    auto Rc = op::slice(R).add_dim(2 * hidden_size, 3 * hidden_size).add_dim(0, hidden_size);
    auto Bc = op::slice(B).add_dim(2 * hidden_size, 3 * hidden_size);
    auto Tc = op::dot(Xt, op::transpose(Wc)) + op::dot(Ht_1, op::transpose(Rc)) + Bc;
    auto ct = clip_activation(activation_g, should_clip, clip, Tc);

    auto Wo = op::slice(W).add_dim(3 * hidden_size, 4 * hidden_size).add_dim(0, input_size);
    auto Ro = op::slice(R).add_dim(3 * hidden_size, 4 * hidden_size).add_dim(0, hidden_size);
    auto Bo = op::slice(B).add_dim(3 * hidden_size, 4 * hidden_size);
    auto To = op::dot(Xt, op::transpose(Wo)) + op::dot(Ht_1, op::transpose(Ro)) + Bo;
    auto ot = clip_activation(activation_f, should_clip, clip, To);

    auto Ct = ft * Ct_1 + it * ct;
    auto Ht = ot * clip_activation(activation_h, should_clip, clip, Ct);

    return edsl::make_tuple(Ct, Ht);
  });
}

}  // namespace PlaidMLPlugin
