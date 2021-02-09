// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset3.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerShapeOf() {
  registerOp("ShapeOf", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset3::ShapeOf>(ctx.layer);
    auto edsl_shape = ctx.operands.at(0).compute_shape();
    DType type = to_plaidml(layer->get_output_type());
    std::vector<int64_t> dims(1, edsl_shape.rank());
    TensorShape ts(type, dims);
    Buffer buffer(ts);
    buffer.copy_from(edsl_shape.sizes().data());
    return edsl::make_tuple(edsl::Constant(buffer, ctx.layer->get_friendly_name()));
  });
}

}  // namespace PlaidMLPlugin
