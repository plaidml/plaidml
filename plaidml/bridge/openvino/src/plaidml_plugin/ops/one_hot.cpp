// Copyright (C) 2020 Intel Corporation
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

static OpRegistration reg("onehot", [](const Context& ctx) {
  IE_ASSERT(ctx.operands.size() == 4);
  auto indices = ctx.operands.at(0);
  auto depth = get_shape_from_constant_operand(1, ctx.layer);  // ctx.operands.at(1);
  auto on_value = ctx.operands.at(2);
  auto off_value = ctx.operands.at(3);
  auto* layer = dynamic_cast<ngraph::opset1::OneHot*>(ctx.layer);
  auto axis = size_t(layer->get_axis());

  std::vector<edsl::TensorDim> I_dims(indices.rank());
  indices.bind_dims(I_dims);
  std::vector<edsl::TensorDim> O_dims(indices.rank() + 1);

  if (axis < 0) axis = indices.rank() + axis;
  size_t j = 0;
  for (size_t i = 0; i < O_dims.size(); i++) {
    if (i == axis) {
      O_dims[i] = edsl::TensorDim(depth[0]);
    } else {
      O_dims[i] = I_dims[j];
      j++;
    }
  }

  edsl::Tensor O = edsl::TensorOutput(O_dims);
  std::vector<edsl::TensorIndex> O_idxs(indices.rank() + 1);
  std::vector<edsl::TensorIndex> I_idxs(indices.rank());
  edsl::Tensor count = edsl::index(O_dims, axis);
  edsl::TensorIndex c;
  O(O_idxs) = indices(I_idxs) == count(c);
  O = select(O, on_value, off_value);
  return edsl::make_tuple(O);
});

}  // namespace PlaidMLPlugin
