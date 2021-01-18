// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerReorgYolo() {
  registerOp("ReorgYolo", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::ReorgYolo>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 1);
    auto I = ctx.operands.at(0);
    // as openvino op request, the strides have to be divisible by 'H' and 'W'.
    auto strides = layer->get_strides()[0];
    return edsl::make_tuple(op::reorg_yolo(I, strides, true));
  });
}

}  // namespace PlaidMLPlugin
