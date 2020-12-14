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

namespace PlaidMLPlugin {

void registerScatterNDUpdate() {
  registerOp("ScatterNDUpdate", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::ScatterNDUpdate>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 3);
    auto data = ctx.operands.at(0);
    auto indices = ctx.operands.at(1);
    auto updates = ctx.operands.at(2);

    auto O = edsl::scatter(data, indices, updates).mode(edsl::ScatterMode::UPDATE_ND);
    return edsl::make_tuple(O);
  });
}

}  // namespace PlaidMLPlugin
