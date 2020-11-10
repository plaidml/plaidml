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

static OpRegistration reg("elu", [](const Context& ctx) {
  auto* layer = ngraph::as_type<ngraph::opset1::Elu>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  auto alpha = layer->get_alpha();

  return edsl::make_tuple(edsl::select(I >= 0, I, alpha * (edsl::exp(I) - 1)));
});

}  // namespace PlaidMLPlugin
