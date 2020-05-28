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

static OpRegistration reg("relu", [](const Context& ctx) {
  // auto* layer = dynamic_cast<ngraph::opset1::Relu*>(ctx.layer);  // TODO: Will need to recover layer to get alpha
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  // edsl::Tensor alpha(layer->negative_slope);  // TODO: How does nGraph do alpha?
  // return edsl::make_tuple(op::relu(I).alpha(alpha));
  return edsl::make_tuple(op::relu(I));
});

}  // namespace PlaidMLPlugin
