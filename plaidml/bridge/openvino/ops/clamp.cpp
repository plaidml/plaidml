// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using plaidml::edsl::make_tuple;
using plaidml::edsl::Tensor;

namespace PlaidMLPlugin {

void registerClamp() {
  registerOp("Clamp", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset1::Clamp>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 1);
    auto I = ctx.operands.at(0);
    Tensor min(layer->get_min());
    Tensor max(layer->get_max());
    return make_tuple(plaidml::op::clip(I, min, max));
  });
}

}  // namespace PlaidMLPlugin
