// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("gelu", [](const Context& ctx) {
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  auto O = I * 0.5 * (1 + edsl::erf(I / sqrt(2)));
  return edsl::make_tuple(O);
});

}  // namespace PlaidMLPlugin
