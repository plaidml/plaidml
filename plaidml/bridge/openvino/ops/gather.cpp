// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerGather() {
  registerOp("Gather", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::Gather>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 3);
    auto I = ctx.operands.at(0);
    auto IX = ctx.operands.at(1);

    auto axis = cast_constant_operand<int64_t>(2, layer);
    edsl::Tensor O = edsl::gather(I, IX).axis(static_cast<int>(axis[0]));
    return edsl::make_tuple(O);
  });
}

}  // namespace PlaidMLPlugin
