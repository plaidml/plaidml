// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <math.h>

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset2.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("depthtospace", [](const Context& ctx) {
  auto* layer = ngraph::as_type<ngraph::opset1::DepthToSpace>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  IE_ASSERT(I.rank() >= 3);

  auto block_size = layer->get_block_size();
  bool blocks_first = layer->get_mode() == ngraph::opset1::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;

  std::vector<edsl::TensorDim> I_dims(I.rank());
  std::vector<edsl::TensorIndex> I_idxs(I.rank());
  std::vector<edsl::TensorIndex> block_idxs(I.rank() - 2);
  std::vector<edsl::TensorDim> O_dims;
  std::vector<edsl::TensorIndex> O_idxs;
  std::vector<edsl::Constraint> constraints;

  I.bind_dims(I_dims);

  size_t total_block_size = 1;
  for (size_t i = 0; i < I.rank() - 2; i++) {
    total_block_size *= block_size;
  }
  O_dims.push_back(I_dims[0]);
  O_dims.push_back(I_dims[1] / total_block_size);
  O_idxs.push_back(I_idxs[0]);
  O_idxs.push_back(I_idxs[1]);
  auto I_channel_stride = blocks_first ? I_dims[1] / total_block_size : edsl::TensorDim(1);
  if (!blocks_first) {
    I_idxs[1] = total_block_size * I_idxs[1];
  }
  for (size_t i = 2; i < I.rank(); i++) {
    // This loop deliberately starts at 2, the 0 & 1 entries are intentionally ignored
    // Note that for constructing the batch index for I, we need to reverse the iteration order, as the largest strides
    // correspond to the earliest indexes for this

    I_idxs[1] = I_idxs[1] + I_channel_stride * block_idxs[I.rank() - 1 - i];
    I_channel_stride = block_size * I_channel_stride;
    O_dims.push_back(I_dims[i] * block_size);
    O_idxs.push_back(block_size * I_idxs[i] + block_idxs[i - 2]);
    constraints.push_back(block_idxs[i - 2] < block_size);
  }
  edsl::Tensor O =
      edsl::Contraction().outShape(O_dims).outAccess(O_idxs).assign(I(I_idxs)).add_constraints(constraints);

  return edsl::make_tuple(O);
});

}  // namespace PlaidMLPlugin
