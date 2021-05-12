// Copyright (C) 2020 Intel Corporation
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

Tensor GatherTree(Tensor STEP_IDS, Tensor PARENT_IDX, Tensor MAX_SEQ_LEN, Tensor END_TOKEN) {
  std::vector<int64_t> step_ids_shape = STEP_IDS.compute_shape().sizes();
  std::vector<int64_t> parent_idx_shape = PARENT_IDX.compute_shape().sizes();
  std::vector<int64_t> max_seq_len_shape = MAX_SEQ_LEN.compute_shape().sizes();
  std::vector<int64_t> end_token_shape = END_TOKEN.compute_shape().sizes();

  // Check shape
  if (step_ids_shape.size() != 3) {
    THROW_IE_EXCEPTION << "step_ids rank must be 3";
  }
  int max_time = step_ids_shape[0];
  int batch_size = parent_idx_shape[1];
  int beam_width = step_ids_shape[2];
  if (max_seq_len_shape[0] != batch_size) {
    THROW_IE_EXCEPTION << "max_seq_len and step_ids must have same batch size";
  }
  if (end_token_shape.size() != 0) {
    THROW_IE_EXCEPTION << "end_token must be a scalar";
  }

  Tensor ZERO = edsl::cast(edsl::index({edsl::TensorDim(1)}, 0), STEP_IDS.dtype());
  Tensor ZERO_INT = edsl::cast(ZERO, DType::INT32);
  // Add padding to update the whole PARENT_IDX value.
  Tensor INDEX_TIME =
      edsl::index({edsl::TensorDim(max_time), edsl::TensorDim(batch_size), edsl::TensorDim(beam_width)}, 0);
  Tensor MAX_TIME = edsl::cast(Tensor(max_time), MAX_SEQ_LEN.dtype());
  Tensor MAX_SEQ_LEN_IN_BEAM = op::minimum(MAX_SEQ_LEN, MAX_TIME);
  Tensor PARENT_IDX_FILTER = INDEX_TIME - edsl::reshape(MAX_SEQ_LEN_IN_BEAM, {1, batch_size, 1});
  // Set padding value to unused index.
  Tensor INDEX_BEAM =
      edsl::index({edsl::TensorDim(max_time), edsl::TensorDim(batch_size), edsl::TensorDim(beam_width)}, 2);
  Tensor PARENT_IDX_NEW = edsl::select(PARENT_IDX_FILTER < 0, edsl::cast(PARENT_IDX, DType::INT32), INDEX_BEAM);
  // Update
  std::vector<Tensor> parents;
  Tensor PARENT = op::squeeze(edsl::gather(INDEX_BEAM, ZERO_INT).axis(0), {0});
  for (int i = max_time - 1; i > 0; i--) {
    parents.push_back(op::unsqueeze(PARENT, {0}));
    Tensor PARENT_IDX_S = op::squeeze(edsl::gather(PARENT_IDX_NEW, ZERO_INT + i).axis(0), {0});
    Tensor NEW_PARENT = edsl::gather(PARENT_IDX_S, op::unsqueeze(PARENT, {-1})).mode(edsl::GatherMode::ND).batchDims(1);
    PARENT = NEW_PARENT;
  }
  parents.push_back(op::unsqueeze(PARENT, {0}));
  std::reverse(parents.begin(), parents.end());
  Tensor PARENTS_IDX_U = op::concatenate(parents, 0);
  // Get output value
  Tensor UPDATE = edsl::gather(STEP_IDS, op::unsqueeze(PARENTS_IDX_U, {-1})).mode(edsl::GatherMode::ND).batchDims(2);
  // Change padding to END_TOKEN
  Tensor OUTPUT = edsl::select(PARENT_IDX_FILTER < 0, UPDATE, END_TOKEN);
  // Check the first decoded END_TOKEN on time axis, values are then filled with END_TOKEN
  Tensor FILTER_FIRST;
  if (OUTPUT.dtype() == DType::FLOAT32) {
    float epsilon = 1e-7f;
    FILTER_FIRST = op::abs(edsl::select(op::abs(OUTPUT - END_TOKEN) < epsilon, OUTPUT, ZERO));
  } else {
    FILTER_FIRST = op::abs(edsl::select(OUTPUT == END_TOKEN, OUTPUT, ZERO));
  }
  Tensor FILTER_EXTEND = op::cumsum(FILTER_FIRST, 0);
  Tensor OUTPUT_F = edsl::select(FILTER_EXTEND < op::abs(END_TOKEN), OUTPUT, END_TOKEN);
  return {OUTPUT_F};
}

void registerGatherTree() {
  registerOp("GatherTree", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 4);
    auto STEP_IDS = ctx.operands.at(0);
    auto PARENT_IDX = ctx.operands.at(1);
    auto MAX_SEQ_LEN = ctx.operands.at(2);
    auto END_TOKEN = ctx.operands.at(3);
    edsl::Tensor O = GatherTree(STEP_IDS, PARENT_IDX, MAX_SEQ_LEN, END_TOKEN);
    return edsl::make_tuple(O);
  });
}

}  // namespace PlaidMLPlugin
