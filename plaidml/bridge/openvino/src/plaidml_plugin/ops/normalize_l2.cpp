// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("normalizel2", [](const Context& ctx) {
  IE_ASSERT(ctx.operands.size() == 2);
  auto I = ctx.operands.at(0);
  std::vector<size_t> axes = get_axis_vector_from_constant_operand(1, ctx.layer);
  auto* layer = dynamic_cast<ngraph::opset1::NormalizeL2*>(ctx.layer);
  // check if axes is empty
  if (axes.empty()) {
    return edsl::make_tuple(I / I);
  }
  int axis = axes[0];
  auto eps = layer->get_eps();
  auto eps_mode = layer->get_eps_mode();
  auto X = op::sum((I * I), edsl::make_tuple(axis), 1);
  if (eps_mode == ngraph::op::EpsMode::ADD) {
    X = X + eps;
  } else if (eps_mode == ngraph::op::EpsMode::MAX) {
    X = edsl::select(X < eps, edsl::Tensor{eps}, X);
  }
  auto norm = edsl::sqrt(X);
  auto O = I / norm;
  return edsl::make_tuple(O);
});

}  // namespace PlaidMLPlugin
