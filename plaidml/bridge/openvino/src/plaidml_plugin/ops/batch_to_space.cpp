// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset2.hpp"

#include "plaidml/op/op.h"
#include "pmlc/util/logging.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("batchtospace", [](const Context& ctx) {
  IE_ASSERT(ctx.operands.size() == 4);
  auto I = ctx.operands.at(0);
  auto block_shape = get_shape_from_constant_operand(1, ctx.layer);
  auto crops_begin = get_coords_from_constant_operand(2, ctx.layer);
  auto crops_end = get_coords_from_constant_operand(3, ctx.layer);

  std::vector<edsl::TensorDim> I_dims(I.rank());
  std::vector<edsl::TensorIndex> I_idxs(I.rank());
  std::vector<edsl::TensorIndex> block_idxs(I.rank() - 1);
  std::vector<edsl::TensorDim> O_dims;
  std::vector<edsl::TensorIndex> O_idxs;
  std::vector<edsl::Constraint> constraints;

  I.bind_dims(I_dims);

  size_t total_block_shape = 1;
  for (auto dim : block_shape) {
    total_block_shape *= dim;
  }
  O_dims.push_back(I_dims[0] / total_block_shape);
  O_idxs.push_back(I_idxs[0]);
  auto I_batch_stride = I_dims[0] / total_block_shape;
  for (size_t i = 1; i < I.rank(); i++) {
    // This loop deliberately starts at 1, the 0 entries are intentionally ignored
    // Note that for constructing the batch index for I, we need to reverse the iteration order, as the largest strides
    // correspond to the earliest indexes for this
    I_idxs[0] = I_idxs[0] + I_batch_stride * block_idxs[I.rank() - 1 - i];
    I_batch_stride = I_batch_stride * block_shape[I.rank() - i];
    O_dims.push_back(I_dims[i] * block_shape[i] - crops_begin[i] - crops_end[i]);
    O_idxs.push_back(block_shape[i] * I_idxs[i] + block_idxs[i - 1] - crops_begin[i]);
    constraints.push_back(block_idxs[i - 1] < block_shape[i]);
  }
  edsl::Tensor O =
      edsl::Contraction().outShape(O_dims).outAccess(O_idxs).assign(I(I_idxs)).add_constraints(constraints);

  return edsl::make_tuple(O);
});

}  // namespace PlaidMLPlugin
