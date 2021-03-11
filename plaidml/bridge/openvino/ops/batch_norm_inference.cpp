// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerBatchNormInference() {
  registerOp("BatchNormInference", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset5::BatchNormInference>(ctx.layer);
    if (!layer) {
      THROW_IE_EXCEPTION << "PlaidML plugin currently only supports the opset5 version of BatchNormInference";
    }
    IE_ASSERT(ctx.operands.size() == 5);
    auto I = ctx.operands.at(0);
    auto gamma = ctx.operands.at(1);
    auto beta = ctx.operands.at(2);
    auto mean = ctx.operands.at(3);
    auto variance = ctx.operands.at(4);
    IE_ASSERT(I.rank() >= 2);
    IE_ASSERT(gamma.rank() == 1);
    IE_ASSERT(beta.rank() == 1);
    IE_ASSERT(mean.rank() == 1);
    IE_ASSERT(variance.rank() == 1);
    // Position the gamma/beta/mean/var data at dimension 1 of the input tensor
    // TODO: Simplify with an oplib-level unsqueeze
    while (gamma.rank() < I.rank() - 1) {
      gamma = op::unsqueeze(gamma, {1});
      beta = op::unsqueeze(beta, {1});
      mean = op::unsqueeze(mean, {1});
      variance = op::unsqueeze(variance, {1});
    }
    return edsl::make_tuple(gamma * ((I - mean) / edsl::sqrt(variance + layer->get_eps_value())) + beta);
  });
}

}  // namespace PlaidMLPlugin
