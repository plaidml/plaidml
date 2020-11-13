// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using namespace plaidml::edsl;

namespace PlaidMLPlugin {

static OpRegistration reg("EmbeddingBagPackedSum", [](const Context& ctx) {
  auto* layer = ngraph::as_type<ngraph::opset4::EmbeddingBagPackedSum>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 2 || ctx.operands.size() == 3);
  auto I = ctx.operands.at(0);
  auto indices = ctx.operands.at(1);
  IE_ASSERT(indices.rank() == 2);

  Tensor per_sample_weights;
  bool with_weights = false;

  if (ctx.operands.size() == 3) {
    per_sample_weights = ctx.operands.at(2);
    IE_ASSERT(per_sample_weights.rank() == 2);
    with_weights = true;
  }

  auto I_gathered = gather(I, indices);
  if (with_weights) {
    std::vector<int64_t> unsqueeze_axes;
    for (int64_t i = per_sample_weights.rank(); i < I_gathered.rank(); i++) {
      unsqueeze_axes.push_back(i);
    }
    auto weights_expanded = op::unsqueeze(per_sample_weights, unsqueeze_axes);
    I_gathered = I_gathered * weights_expanded;
  }
  auto reduced = op::sum(I_gathered, edsl::make_tuple(1), false);
  return edsl::make_tuple(reduced);
});

}  // namespace PlaidMLPlugin
