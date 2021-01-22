// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;  // NOLINT[build/namespaces]
using ngraph::opset1::ReverseSequence;

namespace {

template <typename T>
std::vector<T> cast_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto* ngraph_const = ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(operand_idx));
  if (ngraph_const) {
    return ngraph_const->cast_vector<T>();
  } else {
    THROW_IE_EXCEPTION << " input [1] is Unsupported inputType; ";
  }
}

edsl::Tensor reverse_tensor(edsl::Tensor reverse_crop, int64_t seq_axis) {
  std::vector<edsl::TensorDim> dims(reverse_crop.rank());
  reverse_crop.bind_dims(dims);
  std::vector<edsl::TensorIndex> I_idxs(reverse_crop.rank());
  std::vector<edsl::TensorIndex> O_idxs = I_idxs;
  O_idxs[seq_axis] = dims[seq_axis] - 1 - I_idxs[seq_axis];
  return edsl::Contraction().outShape(dims).outAccess(O_idxs).assign(reverse_crop(I_idxs));
}

}  // namespace

namespace PlaidMLPlugin {

void registerReverseSequence() {
  registerOp("ReverseSequence", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ReverseSequence>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);

    auto batch_axis = layer->get_origin_batch_axis();
    auto seq_axis = layer->get_origin_sequence_axis();
    auto length = cast_constant_operand<int64_t>(1, layer);

    auto shapes = I.compute_shape().sizes();
    std::vector<edsl::Tensor> slice_pools;
    for (int64_t i = 0; i < shapes[batch_axis]; i++) {
      auto batch_indices = edsl::index({edsl::TensorDim(1)}, 0) + i;
      auto I_slice = edsl::gather(I, batch_indices).axis(batch_axis);
      auto indices_reverse = edsl::index({edsl::TensorDim(length[i])}, 0);
      auto I_reverse = edsl::gather(I_slice, indices_reverse).axis(seq_axis);
      auto reverse_crop = reverse_tensor(I_reverse, seq_axis);
      if (length[i] < shapes[seq_axis]) {
        auto indices_constant = edsl::index({edsl::TensorDim(shapes[seq_axis] - length[i])}, 0) + length[i];
        auto I_constant = edsl::gather(I_slice, indices_constant).axis(seq_axis);
        // reverse and concatenate.
        slice_pools.push_back(op::concatenate({reverse_crop, I_constant}, seq_axis));
      } else {
        slice_pools.push_back(reverse_crop);
      }
    }

    auto O = op::concatenate(slice_pools, batch_axis);

    return edsl::make_tuple(O);
  });
}

}  // namespace PlaidMLPlugin
