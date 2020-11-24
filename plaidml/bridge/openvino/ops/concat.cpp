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

static OpRegistration reg("concat", [](const Context& ctx) {
  auto* layer = ngraph::as_type<ngraph::opset1::Concat>(ctx.layer);
  IE_ASSERT(ctx.operands.size() >= 1);
  return edsl::make_tuple(op::concatenate(ctx.operands, layer->get_axis()));
});

}  // namespace PlaidMLPlugin
