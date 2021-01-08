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

void registerActivations() {
  registerOp("elu", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset1::Elu>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 1);
    auto I = ctx.operands.at(0);
    auto alpha = layer->get_alpha();
    return edsl::make_tuple(edsl::select(I >= 0, I, edsl::cast(alpha * (edsl::exp(I) - 1), I.dtype())));
  });

  registerOp("gelu", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 1);
    auto I = ctx.operands.at(0);
    auto O = I * 0.5 * (1 + edsl::erf(I / sqrt(2)));
    return edsl::make_tuple(O);
  });

  registerOp("prelu", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    auto slope = ctx.operands.at(1);
    auto O = select(I < 0.0, edsl::cast(slope * I, I.dtype()), I);
    return edsl::make_tuple(O);
  });

  registerOp("relu", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 1);
    auto I = ctx.operands.at(0);
    return edsl::make_tuple(op::relu(I));
  });

  registerOp("selu", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 3);
    auto I = ctx.operands.at(0);
    auto alpha = ctx.operands.at(1);
    auto lambda = ctx.operands.at(2);
    return edsl::make_tuple(lambda * edsl::select(I > 0, I, edsl::cast(alpha * (edsl::exp(I) - 1), I.dtype())));
  });
}

}  // namespace PlaidMLPlugin
