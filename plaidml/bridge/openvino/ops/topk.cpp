// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using Direction = edsl::SortDirection;
using SortType = op::TopKSortType;
using Mode = ngraph::opset4::TopK::Mode;
using Sort = ngraph::opset4::TopK::SortType;

namespace PlaidMLPlugin {

void registerTopK() {
  registerOp("topk", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::TopK>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    auto K = layer->get_k();
    auto axis = layer->get_axis();
    auto mode = layer->get_mode();
    auto sort = layer->get_sort_type();

    Direction direction;
    switch (mode) {
      case Mode::MAX:
        direction = Direction::DESC;
        break;
      case Mode::MIN:
        direction = Direction::ASC;
        break;
      default:
        THROW_IE_EXCEPTION << "invalid topk sort mode";
    }

    SortType sort_type;
    switch (sort) {
      case Sort::SORT_VALUES:
        sort_type = SortType::VALUE;
        break;
      case Sort::SORT_INDICES:
        sort_type = SortType::INDEX;
        break;
      case Sort::NONE:
        // TODO: According to OV specs, the behavior of Sort::NONE is undefined.
        THROW_IE_EXCEPTION << "SortType::NONE is not implemented";
        break;
    }

    auto O = op::topk(I, K).axis(axis).sort_direction(direction).sort_type(sort_type).build();

    return edsl::make_tuple(O);
  });
}

}  // namespace PlaidMLPlugin
