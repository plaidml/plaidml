// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using edsl::Tensor;

namespace PlaidMLPlugin {

Tensor GatherTree(Tensor STEP_IDS, Tensor PARENT_IDX, Tensor MAX_SEQ_LEN, std::vector<int32_t> max_seq_len,
                  Tensor END_TOKEN) {
  std::vector<int64_t> step_ids_shape = STEP_IDS.compute_shape().sizes();
  std::vector<int64_t> parent_idx_shape = PARENT_IDX.compute_shape().sizes();
  std::vector<int64_t> max_seq_len_shape = MAX_SEQ_LEN.compute_shape().sizes();
  std::vector<int64_t> end_token_shape = END_TOKEN.compute_shape().sizes();

  // Compare Shape

  std::cout << "step_ids:" << STEP_IDS.compute_shape().str() << std::endl;
  std::cout << "parent_idx:" << PARENT_IDX.compute_shape().str() << std::endl;
  std::cout << "max_seq_len:" << MAX_SEQ_LEN.compute_shape().str() << std::endl;
  std::cout << "end_token:" << END_TOKEN.compute_shape().str() << std::endl;
  std::cout << "end_token:" << end_token_shape.size() << std::endl;
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

  // Tensor INDEX_PARENT = edsl::index({edsl::TensorDim(max_time), edsl::TensorDim(batch_size),
  // edsl::TensorDim(beam_width)}, 1); Tensor PARENT_IDX_NEW =  op::concatenate(
  //    {
  //      op::unsqueeze(INDEX_PARENT, {-1}),
  //      op::unsqueeze(PARENT_IDX, {-1})
  //    },
  //    -1
  //);

  // Tensor OUTPUT = op::broadcast(END_TOKEN, {max_time, batch_size, beam_width}, {0}); // max_time * batch_size *
  // beam_width
  Tensor OUTPUT_BATCH = op::broadcast(edsl::reshape(END_TOKEN, {edsl::TensorDim(1)}), {max_time, beam_width},
                                      {0});  // max_time * batch_size * beam_width
  // Tensor MAX_SEQUENCE_IN_BEAM_A = op::minimum(MAX_SEQ_LEN, Tensor(max_time));
  auto INDEX = edsl::index({edsl::TensorDim(beam_width)}, 0);
  Tensor INDEX_M = op::unsqueeze(edsl::index({edsl::TensorDim(max_time), edsl::TensorDim(beam_width)}, 0), {-1});
  // auto INDEX1 = edsl::index({edsl::TensorDim(max_time)}, 0);
  std::vector<Tensor> outputs;
  for (int batch = 0; batch < batch_size; batch++) {
    int max_sequence_in_beam = std::min(max_time, max_seq_len[batch]);
    std::cout << "max_seq_len[batch]:" << max_seq_len[batch] << std::endl;
    std::cout << "max_sequence_in_beam:" << max_sequence_in_beam << std::endl;
    if (max_sequence_in_beam == 0) {
      outputs.push_back(edsl::reshape(OUTPUT_BATCH, {max_time, 1, beam_width}));
      continue;
    }
    std::vector<Tensor> parents;
    for (int i = max_sequence_in_beam - 1; i < max_time; i++) {
      parents.push_back(INDEX);
    }
    Tensor PARENT = edsl::reshape(op::slice(PARENT_IDX)
                                      .add_dim(max_sequence_in_beam - 1, max_sequence_in_beam)
                                      .add_dim(batch, batch + 1)
                                      .add_dim(0, beam_width),
                                  {edsl::TensorDim(beam_width)});
    if (max_sequence_in_beam > 1) {
      parents.push_back(PARENT);
    }
    for (int i = max_sequence_in_beam - 2; i > 0; i--) {
      Tensor PARENT_IDX_S = op::slice(PARENT_IDX).add_dim(i, i + 1).add_dim(batch, batch + 1).add_dim(0, beam_width);
      Tensor NEW_PARENT = edsl::reshape(edsl::gather(PARENT_IDX_S, PARENT).axis(2), {edsl::TensorDim(beam_width)});
      parents.push_back(NEW_PARENT);
      PARENT = NEW_PARENT;
    }
    std::reverse(parents.begin(), parents.end());
    Tensor PARENTS = op::concatenate(parents, 0);
    PARENTS = edsl::reshape(PARENTS, {max_time, beam_width, 1});
    Tensor PARENTS_ND = op::concatenate({INDEX_M, PARENTS}, -1);

    Tensor STEP =
        edsl::reshape(op::slice(STEP_IDS).add_dim(0, max_time).add_dim(batch, batch + 1).add_dim(0, beam_width),
                      {max_time, beam_width, 1});  // 3*3
    Tensor UPDATE = edsl::gather(STEP, edsl::cast(PARENTS_ND, DType::INT32)).mode(edsl::GatherMode::ND);
    // Change some value to end_token
    // auto INDEX1 = edsl::index({edsl::TensorDim(max_sequence_in_beam)}, 0);
    // Tensor OUTPUT_CB = edsl::reshape(
    //    edsl::scatter(OUTPUT_BATCH, INDEX1, UPDATE).mode(ScatterMode::UPDATE_SLICE),
    //    {max_time, 1, beam_width}
    //);

    Tensor OUTPUT_CB = edsl::reshape(
        edsl::select(INDEX_M > edsl::cast(Tensor(max_sequence_in_beam - 1), UPDATE.dtype()), END_TOKEN, UPDATE),
        {max_time, 1, beam_width});
    outputs.push_back(OUTPUT_CB);
  }
  Tensor OUTPUT = op::concatenate(outputs, 1);
  // float epsilon = 1e-6f;
  Tensor ZERO = cast(Tensor(0), OUTPUT.dtype());
  Tensor FILTER_A = edsl::select(OUTPUT == END_TOKEN, OUTPUT, ZERO);
  Tensor FILTER_B = op::cumsum(FILTER_A, 0);
  Tensor OUTPUT_F = edsl::select(FILTER_B < END_TOKEN, OUTPUT, END_TOKEN);

  return {OUTPUT_F};
}

void registerGatherTree() {
  registerOp("GatherTree", [](const Context& ctx) {
    // auto* layer = ngraph::as_type<ngraph::opset4::GatherTree>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 4);
    auto STEP_IDS = ctx.operands.at(0);
    auto PARENT_IDX = ctx.operands.at(1);
    auto MAX_SEQ_LEN = ctx.operands.at(2);
    auto END_TOKEN = ctx.operands.at(3);

    auto* max_seq_len_op = ngraph::as_type<ngraph::op::Constant>(ctx.layer->get_input_node_ptr(2));
    if (max_seq_len_op == nullptr) {
      THROW_IE_EXCEPTION << "Dynamic output size for GatherTree not supported by PlaidML plugin now";
    }
    std::vector<int32_t> max_seq_len = max_seq_len_op->cast_vector<int32_t>();

    edsl::Tensor O = GatherTree(STEP_IDS, PARENT_IDX, MAX_SEQ_LEN, max_seq_len, END_TOKEN);

    return edsl::make_tuple(O);
  });
}

}  // namespace PlaidMLPlugin
