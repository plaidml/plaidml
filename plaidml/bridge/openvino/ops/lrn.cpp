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

static OpRegistration reg("lrn", [](const Context& ctx) {
  auto* layer = ngraph::as_type<ngraph::opset1::LRN>(ctx.layer);
  auto I = ctx.operands.at(0);
  // Note: The reference implementation and documentation do not appear to agree on whether alpha gets divided by the
  // window size; this matches the reference implementation.
  return edsl::make_tuple(op::lrn(I, {static_cast<int64_t>(layer->get_nsize())})
                              .alpha(layer->get_alpha() / layer->get_nsize())
                              .beta(layer->get_beta())
                              .epsilon(layer->get_bias())
                              .axes({1}));
});

}  // namespace PlaidMLPlugin
