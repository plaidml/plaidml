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

void registerSqueeze() {
  registerOp("squeeze", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    auto axes = get_axis_set_from_constant_operand(1, ctx.layer);
    std::vector<int> v_axes;
    v_axes.assign(axes.begin(), axes.end());
    return edsl::make_tuple(op::squeeze(I, v_axes));
  });
}

}  // namespace PlaidMLPlugin
