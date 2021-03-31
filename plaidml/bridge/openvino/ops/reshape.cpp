// Copyright (C) 2021 Intel Corporation
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

void registerReshape() {
  registerOp("reshape", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset1::Reshape>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    // operands.at(1) is unused, just read the Constant instead
    std::vector<int64_t> shape = cast_constant_operand<int64_t>(1, ctx.layer);

    auto special_zero = layer->get_special_zero();
    if (!special_zero) {
      for (auto dim : shape) {
        if (dim == 0) {
          THROW_IE_EXCEPTION << "Cannot use size 0 dim in reshape with special_zero set to false";
        }
      }
    }

    return edsl::make_tuple(op::reshape(I, edsl::make_tuple<int64_t>(shape)));
  });
}

}  // namespace PlaidMLPlugin
