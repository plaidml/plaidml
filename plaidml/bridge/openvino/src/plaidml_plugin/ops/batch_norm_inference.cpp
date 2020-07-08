// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("batchnorminference", [](const Context& ctx) {
  // TODO: The order of inputs does not match the documentation here:
  //   https://docs.openvinotoolkit.org/latest/_docs_ops_normalization_BatchNormInference_1.html
  // This is because the tests show that the input is expected to be the third input tensor. Presumably either the OV
  // code or the OV docs will eventually be updated to make these consistent, and when that happens we'll need to update
  // this code.
  auto* layer = dynamic_cast<ngraph::opset1::BatchNormInference*>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 5);
  auto I = ctx.operands.at(2);
  auto gamma = ctx.operands.at(0);
  auto beta = ctx.operands.at(1);
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

}  // namespace PlaidMLPlugin
