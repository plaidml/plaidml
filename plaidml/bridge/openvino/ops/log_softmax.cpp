// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerLogSoftmax() {
  registerOp("LogSoftmax", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 1);
    auto I = ctx.operands.at(0);
    auto* layer = ngraph::as_type<ngraph::opset5::LogSoftmax>(ctx.layer);
    auto axis = layer->get_axis();
    return edsl::make_tuple(edsl::log(op::softmax(I, axis)));
  });
}

}  // namespace PlaidMLPlugin
