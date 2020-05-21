// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

IE_SUPPRESS_DEPRECATED_START

static OpRegistration reg("relu", [](const Context& ctx) {
  auto* layer = dynamic_cast<ReLULayer*>(ctx.layer);
  assert(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  edsl::Tensor alpha(layer->negative_slope);
  return edsl::make_tuple(op::relu(I).alpha(alpha));
});

IE_SUPPRESS_DEPRECATED_END

}  // namespace PlaidMLPlugin
