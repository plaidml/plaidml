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
  auto axes = get_axis_vector_from_constant_operand(1, ctx.layer);
  auto* layer = ngraph::as_type<ngraph::opset1::NormalizeL2>(ctx.layer);
  auto eps = layer->get_eps();
  auto eps_mode = layer->get_eps_mode();

  if (axes.empty()) {
    auto N = I + eps;
    return edsl::make_tuple(I / N);
  }

  plaidml::op::EpsMode edsl_eps_mode;
  switch (eps_mode) {
    case ngraph::op::EpsMode::ADD:
      edsl_eps_mode = plaidml::op::EpsMode::ADD;
      break;
    case ngraph::op::EpsMode::MAX:
      edsl_eps_mode = plaidml::op::EpsMode::MAX;
      break;
    default:
      THROW_IE_EXCEPTION << "Invalid eps_mode";
  }
  std::vector<int64_t> axes_l2norm;
  axes_l2norm.assign(axes.begin(), axes.end());
  auto N = op::l2norm(I, axes_l2norm).epsilon(eps).eps_mode(edsl_eps_mode);
  return edsl::make_tuple(I / N);
});

}  // namespace PlaidMLPlugin
