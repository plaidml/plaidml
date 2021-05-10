// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using edsl::Tensor;

namespace PlaidMLPlugin {

Tensor GatherTree(Tensor STEP_IDS, Tensor PARENT_IDX, Tensor MAX_SEQ_LEN, Tensor END_TOKEN) {
  std::vector<int64_t> step_ids_shape = STEP_IDS.compute_shape().sizes();
  std::vector<int64_t> parent_idx_shape = PARENT_IDX.compute_shape().sizes();
  std::vector<int64_t> max_seq_len_shape = MAX_SEQ_LEN.compute_shape().sizes();
  std::vector<int64_t> end_token_shape = END_TOKEN.compute_shape().sizes();

  // Compare Shape
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

  // Add padding to convert dynamic count gather to fixed count gather
  Tensor INDEX_BEAM =
      edsl::index({edsl::TensorDim(max_time), edsl::TensorDim(batch_size), edsl::TensorDim(beam_width)}, 2);
  Tensor INDEX_TIME =
      edsl::index({edsl::TensorDim(max_time), edsl::TensorDim(batch_size), edsl::TensorDim(beam_width)}, 0);
  Tensor MAX_TIME = edsl::cast(Tensor(max_time), MAX_SEQ_LEN.dtype());
  Tensor MAX_SEQ_LEN_IN_BEAM = op::minimum(MAX_SEQ_LEN, MAX_TIME);
  Tensor PARENT_IDX_FILTER = INDEX_TIME - edsl::reshape(MAX_SEQ_LEN_IN_BEAM, {1, batch_size, 1});
  Tensor ZERO = edsl::cast(edsl::index({edsl::TensorDim(1)}, 0), STEP_IDS.dtype());
  Tensor ZERO_INT = edsl::cast(ZERO, DType::INT32);
  Tensor INDEX_BB = edsl::gather(INDEX_BEAM, ZERO_INT).axis(0);
  // The slice behind boundary with keep original sequence
  Tensor PARENT_IDX_NEW = edsl::select(PARENT_IDX_FILTER < 0, PARENT_IDX, edsl::cast(INDEX_BEAM, PARENT_IDX.dtype()));
  Tensor INDEX_BATCH = edsl::index(
      {edsl::TensorDim(max_time), edsl::TensorDim(batch_size), edsl::TensorDim(beam_width), edsl::TensorDim(1)}, 1);
  Tensor PARENT_IDX_COR =
      edsl::cast(op::concatenate({INDEX_BATCH, op::unsqueeze(PARENT_IDX_NEW, {-1})}, -1), DType::INT32);
  // Update index
  std::vector<Tensor> parents;
  Tensor INDEX_A = edsl::index({edsl::TensorDim(batch_size), edsl::TensorDim(beam_width), edsl::TensorDim(1)}, 0);
  Tensor INDEX_B = edsl::index({edsl::TensorDim(batch_size), edsl::TensorDim(beam_width), edsl::TensorDim(1)}, 1);
  Tensor INDEX_O = op::concatenate({INDEX_A, INDEX_B}, -1);
  Tensor PARENT = INDEX_O;
  for (int i = max_time - 1; i >= 0; i--) {
    parents.push_back(op::unsqueeze(PARENT, {0}));
    Tensor PARENT_IDX_S = op::squeeze(edsl::gather(PARENT_IDX_COR, ZERO_INT + i).axis(0), {0});
    Tensor NEW_PARENT =
        op::squeeze(edsl::gather(op::unsqueeze(PARENT_IDX_S, {-1}), PARENT).mode(edsl::GatherMode::ND), {-1});
    PARENT = NEW_PARENT;
  }
  std::reverse(parents.begin(), parents.end());
  Tensor PARENTS_ND = op::concatenate(parents, 0);
  Tensor INDEX_ND = edsl::index(
      {edsl::TensorDim(max_time), edsl::TensorDim(batch_size), edsl::TensorDim(beam_width), edsl::TensorDim(2)}, 0);
  Tensor PARENTS_ND_FILTER = INDEX_ND - edsl::reshape(MAX_SEQ_LEN_IN_BEAM, {1, batch_size, 1, 1});
  Tensor PARENTS_ND_T =
      edsl::select(PARENTS_ND_FILTER < 0, PARENTS_ND, op::unsqueeze(edsl::cast(INDEX_O, PARENTS_ND.dtype()), {0}));
  Tensor PARENTS_ND_NEW = op::concatenate({op::unsqueeze(INDEX_TIME, {-1}), PARENTS_ND_T}, -1);
  // Get output value
  Tensor UPDATE = edsl::gather(op::unsqueeze(STEP_IDS, {-1}), PARENTS_ND_NEW).mode(edsl::GatherMode::ND);
  // Change padding to END_TOKEN
  Tensor OUTPUT = edsl::select(PARENT_IDX_FILTER < 0, op::squeeze(UPDATE, {-1}), END_TOKEN);

  // Check the first decoded END_TOKEN on time axis, values are then filled with END_TOKEN
  Tensor FILTER_A;
  if (OUTPUT.dtype() == DType::FLOAT32) {
    float epsilon = 1e-7f;
    FILTER_A = op::abs(edsl::select(op::abs(OUTPUT - END_TOKEN) < epsilon, OUTPUT, ZERO));
  } else {
    FILTER_A = op::abs(edsl::select(OUTPUT == END_TOKEN, OUTPUT, ZERO));
  }
  Tensor FILTER_B = op::cumsum(FILTER_A, 0);
  Tensor OUTPUT_F = edsl::select(FILTER_B < op::abs(END_TOKEN), OUTPUT, END_TOKEN);
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
