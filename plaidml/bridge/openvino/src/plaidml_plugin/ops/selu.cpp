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

static OpRegistration reg("selu", [](const Context& ctx) {
  IE_ASSERT(ctx.operands.size() == 3);
  auto I = ctx.operands.at(0);
  auto alpha = ctx.operands.at(1);
  auto lambda = ctx.operands.at(2);

  return edsl::make_tuple(lambda * edsl::select(I > 0, I, alpha * (edsl::exp(I) - 1)));
});

}  // namespace PlaidMLPlugin
