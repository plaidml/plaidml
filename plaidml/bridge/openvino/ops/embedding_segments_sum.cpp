// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace {

template <typename T>
std::vector<T> cast_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto* ngraph_const = ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(operand_idx));
  if (ngraph_const) {
    return ngraph_const->cast_vector<T>();
  } else {
    THROW_IE_EXCEPTION << "Dynamic slicing not currently supported by PlaidML plugin; all of indices, segment_ids, "
                          "num_segments and default index must be Constants.";
  }
}

}  // namespace

namespace PlaidMLPlugin {

void registerEmbeddingSegmentsSum() {
  registerOp("EmbeddingSegmentsSum", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::EmbeddingSegmentsSum>(ctx.layer);
    IE_ASSERT(ctx.operands.size() >= 4);
    IE_ASSERT(ctx.operands.size() <= 6);
    auto I = ctx.operands.at(0);
    auto indices = ctx.operands.at(1);
    IE_ASSERT(indices.rank() == 1);
    auto segment_ids = ctx.operands.at(2);
    auto num_segments = cast_constant_operand<int32_t>(3, layer)[0];
    int32_t default_index;
    edsl::Tensor per_sample_weights;
    bool with_default_index = false;
    bool with_weights = false;

    if (ctx.operands.size() >= 5) {
      with_default_index = true;
      default_index = cast_constant_operand<int32_t>(4, layer)[0];
      if (ctx.operands.size() == 6) {
        per_sample_weights = ctx.operands.at(5);
        with_weights = true;
      }
    }

    edsl::Tensor I_gathered = edsl::gather(I, indices);

    auto ndims = I_gathered.rank();
    std::vector<edsl::TensorDim> I_dims(ndims);
    std::vector<edsl::TensorIndex> I_idxs(ndims);
    I_gathered.bind_dims(I_dims);
    auto O_dims = I_dims;
    auto O_idxs = I_idxs;

    // Create zero-initialized input tensor.
    O_dims[0] = edsl::TensorDim(num_segments);
    edsl::Tensor scatter_in_shape = edsl::Contraction(O_dims, O_idxs).assign(I(I_idxs));
    auto zero = cast(edsl::Tensor{0}, I_gathered.dtype());
    auto scatter_shape_vec = scatter_in_shape.compute_shape().sizes();
    std::vector<int> scatter_shape(begin(scatter_shape_vec), end(scatter_shape_vec));
    std::vector<int> target_axes = {};
    auto I_scatter_init = op::broadcast(zero, scatter_shape, target_axes);

    if (with_weights) {
      std::vector<int64_t> unsqueeze_axes;
      for (int64_t i = 1; i < I_gathered.rank(); i++) {
        unsqueeze_axes.push_back(i);
      }
      auto weights_expanded = op::unsqueeze(per_sample_weights, unsqueeze_axes);
      I_gathered = I_gathered * weights_expanded;
    }

    edsl::Tensor scattered = edsl::scatter(I_scatter_init, segment_ids, I_gathered);
    if (with_default_index) {
      // Fill empty segments with default slice.
      O_dims[0] = edsl::TensorDim(1);
      O_idxs[0] = I_idxs[0] - default_index;
      edsl::Tensor default_slice = edsl::Contraction(O_dims, O_idxs).assign(I(I_idxs));
      edsl::Tensor I_default = op::repeat(default_slice).count(num_segments).axis(0);
      auto slice_shape = I_gathered.compute_shape().sizes();
      std::vector<int> target_shape(begin(slice_shape), end(slice_shape));
      auto I_zero_slice = op::broadcast(zero, target_shape, target_axes);
      edsl::Tensor I_default_slice =
          edsl::scatter(I_default, segment_ids, I_zero_slice).mode(edsl::ScatterMode::UPDATE_SLICE);
      scattered = scattered + I_default_slice;
    }

    return edsl::make_tuple(scattered);
  });
}

}  // namespace PlaidMLPlugin
