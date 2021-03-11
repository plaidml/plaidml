// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <math.h>

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset2.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerBroadcast() {
  registerOp("Broadcast", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset1::Broadcast>(ctx.layer);
    if (!layer) {
      THROW_IE_EXCEPTION << "PlaidML plugin currently only supports the opset1 version of Broadcast";
    }
    IE_ASSERT(ctx.operands.size() <= 3);
    IE_ASSERT(ctx.operands.size() >= 2);
    auto I = ctx.operands.at(0);
    auto target_shape = get_shape_from_constant_operand(1, layer);
    ngraph::AxisVector axes_mapping;
    auto mode = layer->get_broadcast_spec();
    switch (mode.m_type) {
      case ngraph::op::AutoBroadcastType::EXPLICIT:
        if (ctx.operands.size() != 3) {
          THROW_IE_EXCEPTION << "An axes_mapping must be passed to broadcast when using explicit mode";
        }
        axes_mapping = get_axis_vector_from_constant_operand(2, layer);
        break;
      case ngraph::op::AutoBroadcastType::NUMPY: {
        auto excess_rank = target_shape.size() - I.rank();
        IE_ASSERT(excess_rank >= 0);
        for (size_t i = 0; i < I.rank(); i++) {
          axes_mapping.push_back(i + excess_rank);
        }
        break;
      }
      default:
        THROW_IE_EXCEPTION << "Unrecognized broadcast type";
    }

    std::vector<int64_t> int_target_shape(target_shape.begin(), target_shape.end());
    std::vector<int64_t> int_axes_mapping(axes_mapping.begin(), axes_mapping.end());
    return edsl::make_tuple(op::broadcast(I, int_target_shape, int_axes_mapping));
  });
}

}  // namespace PlaidMLPlugin
