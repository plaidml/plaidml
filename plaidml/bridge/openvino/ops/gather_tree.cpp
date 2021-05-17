// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using edsl::Tensor;

namespace PlaidMLPlugin {

Tensor GatherTree(Tensor step_ids, Tensor parent_idx, Tensor max_seq_len, Tensor end_token) {
  std::vector<int64_t> StepIdsShape = step_ids.compute_shape().sizes();
  std::vector<int64_t> ParentIdxShape = parent_idx.compute_shape().sizes();
  std::vector<int64_t> MaxSeqLenShape = max_seq_len.compute_shape().sizes();
  std::vector<int64_t> EndTokenShape = end_token.compute_shape().sizes();

  // Check shape
  if (StepIdsShape.size() != 3) {
    THROW_IE_EXCEPTION << "step_ids rank must be 3";
  }
  int MaxTime = StepIdsShape[0];
  int BatchSize = ParentIdxShape[1];
  int BeamWidth = StepIdsShape[2];
  if (MaxSeqLenShape[0] != BatchSize) {
    THROW_IE_EXCEPTION << "max_seq_len and step_ids must have same batch size";
  }
  if (EndTokenShape.size() != 0) {
    THROW_IE_EXCEPTION << "end_token must be a scalar";
  }

  Tensor zero = edsl::cast(edsl::index({edsl::TensorDim(1)}, 0), step_ids.dtype());
  Tensor zero_int = edsl::cast(zero, DType::INT32);
  // Add padding to update the whole PARENT_IDX value.
  Tensor index_time =
      edsl::index({edsl::TensorDim(MaxTime), edsl::TensorDim(BatchSize), edsl::TensorDim(BeamWidth)}, 0);
  Tensor max_time = edsl::cast(Tensor(MaxTime), max_seq_len.dtype());
  Tensor max_seq_len_in_beam = op::minimum(max_seq_len, max_time);
  Tensor parent_idx_filter = index_time - edsl::reshape(max_seq_len_in_beam, {1, BatchSize, 1});
  // Set padding value to unused index.
  Tensor index_beam = edsl::index({edsl::TensorDim(BatchSize), edsl::TensorDim(BeamWidth)}, 1);
  Tensor parent_idx_new =
      edsl::select(parent_idx_filter < 0, edsl::cast(parent_idx, DType::INT32), op::unsqueeze(index_beam, {0}));

  // Update with gather.
  std::vector<Tensor> Parents;
  Tensor parent = index_beam;
  // TODO: replace "for" with scanOp.
  for (int i = MaxTime - 1; i > 0; i--) {
    Parents.push_back(op::unsqueeze(parent, {0}));
    Tensor parent_idx_s = op::squeeze(edsl::gather(parent_idx_new, zero_int + i).axis(0), {0});
    Tensor new_parent = edsl::gather(parent_idx_s, op::unsqueeze(parent, {-1})).mode(edsl::GatherMode::ND).batchDims(1);
    parent = new_parent;
  }
  Parents.push_back(op::unsqueeze(parent, {0}));
  std::reverse(Parents.begin(), Parents.end());
  Tensor parents_idx_u = op::concatenate(Parents, 0);
  // Get output value.
  Tensor update = edsl::gather(step_ids, op::unsqueeze(parents_idx_u, {-1})).mode(edsl::GatherMode::ND).batchDims(2);
  // Change padding to end_token.
  Tensor output = edsl::select(parent_idx_filter < 0, update, end_token);
  // Check the first decoded end_token on time axis, values are then filled with end_token.
  Tensor filter_first;
  if (output.dtype() == DType::FLOAT32) {
    float Epsilon = 1e-7f;
    filter_first = op::abs(edsl::select(op::abs(output - end_token) < Epsilon, output, zero));
  } else {
    filter_first = op::abs(edsl::select(output == end_token, output, zero));
  }
  Tensor filter_extend = op::cumsum(filter_first, 0);
  Tensor output_f = edsl::select(filter_extend < op::abs(end_token), output, end_token);
  return output_f;
}

void registerGatherTree() {
  registerOp("GatherTree", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 4);
    auto step_ids = ctx.operands.at(0);
    auto parent_idx = ctx.operands.at(1);
    auto max_seq_len = ctx.operands.at(2);
    auto end_token = ctx.operands.at(3);
    edsl::Tensor o = GatherTree(step_ids, parent_idx, max_seq_len, end_token);
    return edsl::make_tuple(o);
  });
}

}  // namespace PlaidMLPlugin
