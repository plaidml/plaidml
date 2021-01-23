// Copyright (C) 2021 Intel Corporation
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

namespace {

// TODO: Remove and replace use with get_axis_set_from_constant_operand once upstream fixed for negatives
ngraph::AxisSet cast_constant_operand_to_axis_set(size_t operand_idx, size_t ndims, ngraph::Node* layer) {
  auto* ngraph_const = ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(operand_idx));
  if (ngraph_const) {
    ngraph::AxisSet ret;
    auto axis_vec = ngraph_const->cast_vector<int64_t>();
    for (auto ax : axis_vec) {
      if (ax < 0) {
        ax += ndims;
        if (ax < 0) {
          THROW_IE_EXCEPTION << "Axis underflow in Unsqueeze (requested axis more negative than rank of tensor)";
        }
      }
      ret.emplace(ax);
    }
    return ret;
  } else {
    THROW_IE_EXCEPTION << " input [1] is Unsupported inputType; ";
  }
}

}  // namespace

void registerUnsqueeze() {
  registerOp("unsqueeze", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    auto axes = cast_constant_operand_to_axis_set(1, I.rank(), ctx.layer);
    std::vector<edsl::TensorDim> I_dims(I.rank());
    size_t new_rank = I.rank() + axes.size();
    I.bind_dims(I_dims);
    std::vector<edsl::TensorDim> O_dims;
    size_t src_loc = 0;
    for (size_t dst_loc = 0; dst_loc < new_rank; dst_loc++) {
      if (axes.count(dst_loc)) {
        O_dims.push_back(edsl::TensorDim(1));
      } else {
        O_dims.push_back(I_dims[src_loc]);
        src_loc++;
      }
    }
    IE_ASSERT(src_loc == I.rank());  // Need to have used exactly all the input dims
    return edsl::make_tuple(op::reshape(I, edsl::make_tuple(O_dims)));
  });
}

}  // namespace PlaidMLPlugin
