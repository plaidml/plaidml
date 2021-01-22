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

void registerPad() {
  registerOp("pad", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset1::Pad>(ctx.layer);
    IE_ASSERT((ctx.operands.size() == 3) || (ctx.operands.size() == 4));

    auto I = ctx.operands.at(0);
    auto lo_pads = cast_constant_operand<int>(1, layer);
    auto hi_pads = cast_constant_operand<int>(2, layer);

    auto autopad_mode = to_plaidml(layer->get_pad_mode());

    auto op = op::explicit_padding(I, lo_pads, hi_pads).mode(autopad_mode);

    if (ctx.operands.size() == 4) {
      op.padval(ctx.operands.at(3));
    }

    return edsl::make_tuple(op);
  });
}

}  // namespace PlaidMLPlugin
