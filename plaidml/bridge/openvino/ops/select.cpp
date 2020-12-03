// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerSelect() {
  registerOp("select", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 3);
    auto A = ctx.operands.at(0);
    auto B = ctx.operands.at(1);
    auto C = ctx.operands.at(2);
    DType target_type = to_plaidml(ctx.layer->get_element_type());
    return edsl::make_tuple(select(A, cast(B, target_type), cast(C, target_type)));
  });
}

}  // namespace PlaidMLPlugin
