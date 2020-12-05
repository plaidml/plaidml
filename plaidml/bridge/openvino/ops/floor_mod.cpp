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

void registerFloorMod() {
  registerOp("FloorMod", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto N = ctx.operands.at(0);
    auto D = ctx.operands.at(1);
    return edsl::make_tuple(N - D * edsl::floor(N / D));
  });
}

}  // namespace PlaidMLPlugin
