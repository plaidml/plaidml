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

static OpRegistration reg("fakequantize", [](const Context& ctx) {
  auto* layer = dynamic_cast<ngraph::opset1::FakeQuantize*>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 5);
  auto X = ctx.operands.at(0);
  auto in_lo = ctx.operands.at(1);
  auto in_hi = ctx.operands.at(2);
  auto out_lo = ctx.operands.at(3);
  auto out_hi = ctx.operands.at(4);
  uint64_t levels = layer->get_levels();
  if (levels <= 1) {
    THROW_IE_EXCEPTION << "FakeQuantize requires at least 2 levels";
  }
  auto lvl_diff = levels - 1;
  auto main_result = edsl::round((X - in_lo) / (in_hi - in_lo) * (lvl_diff)) / (lvl_diff) * (out_hi - out_lo) + out_lo;
  return edsl::make_tuple(edsl::select(X <= op::minimum(in_lo, in_hi), out_lo,
                                       edsl::select(X > op::maximum(in_lo, in_hi), out_hi, main_result)));
});

}  // namespace PlaidMLPlugin
