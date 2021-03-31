// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerHardSigmoid() {
  registerOp("HardSigmoid", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 3);
    auto I = ctx.operands.at(0);
    auto alpha = ctx.operands.at(1);
    auto beta = ctx.operands.at(2);
    auto O = edsl::cast(edsl::Tensor(1.0), I.dtype());
    auto Z = edsl::cast(edsl::Tensor(0.0), I.dtype());

    I = alpha * I + beta;
    I = edsl::select(I > O, O, I);
    I = edsl::select(I < Z, Z, I);

    return edsl::make_tuple(I);
  });
}

}  // namespace PlaidMLPlugin
