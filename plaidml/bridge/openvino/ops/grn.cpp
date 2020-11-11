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
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  if (I.rank() > 4 || I.rank() < 2) {
    THROW_IE_EXCEPTION << "input tensor must be 2 <= rank <= 4 ";
  }
  auto* layer = ngraph::as_type<ngraph::opset1::GRN>(ctx.layer);
  auto bias = layer->get_bias();
  auto N = op::l2norm(I, {1}).epsilon(bias);
  return edsl::make_tuple(I / N);
});

}  // namespace PlaidMLPlugin
