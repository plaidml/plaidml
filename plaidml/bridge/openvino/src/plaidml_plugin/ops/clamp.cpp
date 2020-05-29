// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using namespace plaidml; // NOLINT[build/namespaces]
using namespace InferenceEngine; // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("clamp", [](const Context& ctx) {
  auto* layer = dynamic_cast<ngraph::opset1::Clamp*>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  edsl::Tensor min(layer->get_min());
  edsl::Tensor max(layer->get_max());
  return edsl::make_tuple(op::clip(I, min, max));
});

}  // namespace PlaidMLPlugin
