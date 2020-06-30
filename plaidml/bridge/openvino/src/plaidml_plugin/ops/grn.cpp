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

static OpRegistration reg("grn", [](const Context& ctx) {
  // inputs: I (rank 2 to 4 )
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  auto* layer = dynamic_cast<ngraph::opset1::GRN*>(ctx.layer);
  auto bias = layer->get_bias();
  auto X = op::sum((I * I), edsl::make_tuple(1), 1);
  auto t = X + bias;
  auto norm = edsl::sqrt(t);
  auto O = I / norm;
  return edsl::make_tuple(O);
});

}  // namespace PlaidMLPlugin
