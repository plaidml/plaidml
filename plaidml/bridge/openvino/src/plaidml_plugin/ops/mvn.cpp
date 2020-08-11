// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset2.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("mvn", [](const Context& ctx) {
  auto* layer = dynamic_cast<ngraph::opset2::MVN*>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  return edsl::make_tuple(op::mvn(I)
                              .axes(layer->get_reduction_axes().to_vector())
                              .normalize_variance(layer->get_normalize_variance())
                              .epsilon(layer->get_eps()));
});

}  // namespace PlaidMLPlugin
