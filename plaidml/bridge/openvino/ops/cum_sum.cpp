// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset3.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace {

edsl::Tensor reverse_tensor(edsl::Tensor I, int64_t seq_axis) {
  std::vector<edsl::TensorDim> dims(I.rank());
  I.bind_dims(dims);
  std::vector<edsl::TensorIndex> I_idxs(I.rank());
  std::vector<edsl::TensorIndex> O_idxs(I_idxs);
  O_idxs[seq_axis] = dims[seq_axis] - 1 - I_idxs[seq_axis];
  return edsl::Contraction().outShape(dims).outAccess(O_idxs).assign(I(I_idxs));
}

}  // namespace

namespace PlaidMLPlugin {

void registerCumSum() {
  registerOp("CumSum", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    auto* layer = ngraph::as_type<ngraph::opset3::CumSum>(ctx.layer);
    auto axis = cast_constant_operand<int64_t>(1, layer)[0];
    if (axis < 0) {
      axis = axis + I.rank();
    }
    auto is_exclusive = layer->is_exclusive();
    auto is_reverse = layer->is_reverse();

    auto I_reverse = is_reverse ? reverse_tensor(I, axis) : I;
    auto O = op::cumsum(I_reverse, axis, is_exclusive);
    auto O_reverse = is_reverse ? reverse_tensor(O, axis) : O;
    return edsl::make_tuple(O_reverse);
  });
}

}  // namespace PlaidMLPlugin
