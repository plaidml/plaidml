// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerRound() {
  registerOp("Round", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 1);
    auto I = ctx.operands.at(0);
    auto* layer = ngraph::as_type<ngraph::opset5::Round>(ctx.layer);
    auto round_mode = layer->get_mode();

    switch (round_mode) {
      case ngraph::op::v5::Round::RoundMode::HALF_TO_EVEN: {
        auto unsigned_I = op::abs(I);
        auto int_part = edsl::floor(unsigned_I);

        auto zero = edsl::cast(edsl::Tensor{0}, I.dtype());
        auto one = edsl::cast(edsl::Tensor{1}, I.dtype());
        auto minus_one = edsl::cast(edsl::Tensor{-1}, I.dtype());
        auto sign = edsl::select(I >= 0, one, minus_one);
        auto flag = edsl::select((unsigned_I - int_part) == 0.5, one, zero);
        auto offset = int_part % 2 - 1;

        auto O = (edsl::round(unsigned_I) + offset * flag) * sign;
        return edsl::make_tuple(O);
      }
      case ngraph::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO: {
        return edsl::make_tuple(round(I));
      }
      default: {
        throw std::runtime_error("Unsupported Round Mode");
      }
    }
  });
}

}  // namespace PlaidMLPlugin
