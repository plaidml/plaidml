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
    std::vector<edsl::TensorIndex> O_idxs;
    for (int64_t axis = 0; axis < reverse_crop.rank(); axis++) {
        if (axis == seq_axis) {
            O_idxs.push_back(dims[axis] - 1 - I_idxs[axis]);
        } else {
            O_idxs.push_back(I_idxs[axis]);
        }
    }
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
    for (auto len : length) {
      if (len < 1) {
        THROW_IE_EXCEPTION << " input seq len at least be one ";
      }
    }

    auto shapes = I.compute_shape().sizes();
    std::vector<edsl::Tensor> slice_pools;
    for (int64_t i = 0; i < shapes[batch_axis]; i++) {
      // split reversed tensor.
      auto reverse_piece = op::slice(I);
      auto constant_piece = op::slice(I);
      for (int64_t j = 0; j < shapes.size(); j++) {
        if (j == batch_axis) {
          reverse_piece.add_dim(i);
          constant_piece.add_dim(i);
        } else if (j == seq_axis) {
          reverse_piece.add_dim(0, length[i], 1);
          constant_piece.add_dim(length[i], shapes[seq_axis], 1);
        } else {
          reverse_piece.add_dim(0, shapes[j], 1);
          constant_piece.add_dim(0, shapes[j], 1);
        }
      }

      // after slice, one-dim will be cast, then unsqueeze them for concatenate
      edsl::Tensor reverse, constant;
      if (length[i] == 1) {
        if (batch_axis < seq_axis) {
          reverse = op::unsqueeze(op::unsqueeze(reverse_piece, {batch_axis}), {seq_axis});
        } else {
          reverse = op::unsqueeze(op::unsqueeze(reverse_piece, {seq_axis}), {batch_axis});
        }
      } else {
        reverse = op::unsqueeze(reverse_piece, {batch_axis});
      }
      if (length[i] == shapes[seq_axis] - 1) {
        if (batch_axis < seq_axis) {
          constant = op::unsqueeze(op::unsqueeze(constant_piece, {batch_axis}), {seq_axis});
        } else {
          constant = op::unsqueeze(op::unsqueeze(constant_piece, {seq_axis}), {batch_axis});
        }
      } else {
        constant = op::unsqueeze(constant_piece, {batch_axis});
      }

      // reverse and concatenate.
      auto reverse_crop = reverse_tensor(reverse, seq_axis);
      slice_pools.push_back(op::concatenate({reverse_crop, constant}, seq_axis));
    }

    auto O = op::concatenate(slice_pools, batch_axis);

    return edsl::make_tuple(O);
  });
}

}  // namespace PlaidMLPlugin
