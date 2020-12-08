// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace {

template <typename T>
std::vector<T> cast_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto* ngraph_const = ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(operand_idx));
  if (ngraph_const) {
    return ngraph_const->cast_vector<T>();
  } else {
    THROW_IE_EXCEPTION
        << "Dynamic slicing not currently supported by PlaidML plugin; all of indices, offsets and default index"
           "must be Constants.";
  }
}

}  // namespace

namespace PlaidMLPlugin {
void registerGather() {
  registerOp("Gather", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::Gather>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 3);
    auto I = ctx.operands.at(0);
    auto IX = ctx.operands.at(1);

    auto axis = cast_constant_operand<int64_t>(2, layer);
    edsl::Tensor O = edsl::gather(I, IX).axis(static_cast<int>(axis[0]));
    return edsl::make_tuple(O);
  });
}

}  // namespace PlaidMLPlugin
