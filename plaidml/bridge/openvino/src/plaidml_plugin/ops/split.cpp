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

static OpRegistration reg("split", [](const Context& ctx) {
  auto* layer = dynamic_cast<ngraph::opset1::Split*>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 2);
  auto I = ctx.operands.at(0);
  // operands.at(1) is unused, just read the Constant instead
  auto splits = layer->get_num_splits();
  ngraph::AxisSet axes;
  auto axis_ngraph_op = std::dynamic_pointer_cast<ngraph::op::Constant>(layer->input_value(1).get_node_shared_ptr());
  if (axis_ngraph_op) {
    axes = axis_ngraph_op->get_axis_set_val();
  } else {
    THROW_IE_EXCEPTION << "Dynamic axis not currently supported by PlaidML plugin";
  }
  IE_ASSERT(axes.size() == 1);
  auto axis = axes.to_vector()[0];

  auto ndims = I.rank();
  std::vector<edsl::TensorDim> I_dims(ndims);
  std::vector<edsl::TensorIndex> I_idxs(ndims);
  std::vector<edsl::Tensor> Os;
  I.bind_dims(I_dims);
  auto O_dims = I_dims;
  auto split_size = I_dims[axis] / splits;
  O_dims[axis] = split_size;
  for (size_t i = 0; i < splits; i++) {
    auto thisO = edsl::TensorOutput(O_dims);
    auto O_idxs = I_idxs;
    O_idxs[axis] = I_idxs[axis] + i * split_size;
    thisO(O_idxs) = I(I_idxs);
    Os.push_back(thisO);
  }
  return edsl::make_tuple(Os);
});

}  // namespace PlaidMLPlugin
