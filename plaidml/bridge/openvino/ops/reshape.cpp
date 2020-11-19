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

static OpRegistration reg("reshape", [](const Context& ctx) {
  auto* layer = ngraph::as_type<ngraph::opset1::Reshape>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 2);
  auto I = ctx.operands.at(0);
  // operands.at(1) is unused, just read the Constant instead
  ngraph::Shape shape;
  auto shape_ngraph_op = ngraph::as_type_ptr<ngraph::op::Constant>(layer->input_value(1).get_node_shared_ptr());
  if (shape_ngraph_op) {
    shape = shape_ngraph_op->get_shape_val();
  } else {
    THROW_IE_EXCEPTION << "Dynamic reshaping not currently supported by PlaidML plugin";
  }

  auto special_zero = layer->get_special_zero();
  if (!special_zero) {
    for (auto dim : shape) {
      if (dim == 0) {
        THROW_IE_EXCEPTION << "Cannot use size 0 dim in reshape with special_zero set to false";
      }
    }
  }

  return edsl::make_tuple(op::reshape(I, edsl::make_tuple<size_t>(shape)));
});

}  // namespace PlaidMLPlugin
