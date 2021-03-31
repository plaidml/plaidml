// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {
void registerGatherND() {
  registerOp("GatherND", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset5::GatherND>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    auto IX = ctx.operands.at(1);
    auto batchDims = layer->get_batch_dims();

    edsl::Tensor O = edsl::gather(I, IX).mode(edsl::GatherMode::ND).batchDims(batchDims);
    // Reshape leading 'batchDims' into one by multiplying them.
    auto shape = O.compute_shape().sizes();
    size_t firstDim = 1;
    for (size_t i = 0; i < batchDims; ++i) {
      firstDim *= shape.front();
      shape.erase(shape.begin());
    }
    shape.insert(shape.begin(), firstDim);
    O = edsl::reshape(O, shape);

    return edsl::make_tuple(O);
  });
}

}  // namespace PlaidMLPlugin
