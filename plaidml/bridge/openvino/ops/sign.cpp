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

static OpRegistration reg("sign", [](const Context& ctx) {
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  auto Z = edsl::cast(edsl::Tensor(0.0), I.dtype());
  auto O = edsl::cast(edsl::Tensor(1.0), I.dtype());
  return edsl::make_tuple(edsl::select(I > 0, O, Z) + edsl::select(I < 0, -1 * O, Z));
});

}  // namespace PlaidMLPlugin
