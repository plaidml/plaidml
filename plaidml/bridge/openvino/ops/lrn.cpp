// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerLrn() {
  registerOp("lrn", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset1::LRN>(ctx.layer);
    auto I = ctx.operands.at(0);
    auto axes_vec = layer->get_reduction_axes().to_vector();
    size_t axes_num = axes_vec.size();
    int64_t window_size = static_cast<int64_t>(layer->get_nsize());
    // Note: The same 'window_size' applys to all axes.
    std::vector<int64_t> window_size_vec(axes_num, window_size);
    return edsl::make_tuple(op::lrn(I, window_size_vec)
                                .alpha(layer->get_alpha() / std::pow(window_size, axes_num))
                                .beta(layer->get_beta())
                                .epsilon(layer->get_bias())
                                .axes(axes_vec));
  });
}

}  // namespace PlaidMLPlugin
