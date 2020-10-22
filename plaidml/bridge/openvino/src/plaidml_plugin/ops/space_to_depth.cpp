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

static OpRegistration reg("spacetodepth", [](const Context& ctx) {
  auto* layer = dynamic_cast<ngraph::opset1::SpaceToDepth*>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  IE_ASSERT(I.rank() >= 3);

  auto block_size = layer->get_block_size();
  bool blocks_first = layer->get_mode() == ngraph::opset1::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;

  std::vector<edsl::TensorDim> I_dims(I.rank());
  std::vector<edsl::TensorIndex> Reordered_idxs(2 * I.rank() - 2);
  std::vector<edsl::TensorIndex> block_idxs(I.rank() - 2);
  std::vector<edsl::TensorDim> Reordered_dims;
  std::vector<edsl::TensorIndex> I_idxs;
  std::vector<edsl::Constraint> constraints;

  I.bind_dims(I_dims);

  size_t total_block_size = 1;
  for (size_t i = 0; i < I.rank() - 2; i++) {
    total_block_size *= block_size;
  }
  Reordered_dims.push_back(I_dims[0]);
  I_idxs.push_back(Reordered_idxs[0]);
  if (blocks_first) {
    I_idxs.push_back(Reordered_idxs[I.rank() - 1]);
  } else {
    Reordered_dims.push_back(I_dims[1]);
    I_idxs.push_back(Reordered_idxs[1]);
  }
  size_t block_offset = blocks_first ? 1 : 2;
  for (size_t i = 0; i < I.rank() - 2; i++) {
    Reordered_dims.push_back(edsl::TensorDim(block_size));
  }
  if (blocks_first) {
    Reordered_dims.push_back(I_dims[1]);
  }
  for (size_t i = 0; i < I.rank() - 2; i++) {
    Reordered_dims.push_back(I_dims[i + 2] / block_size);
    I_idxs.push_back(block_size * Reordered_idxs[I.rank() + i] + Reordered_idxs[i + block_offset]);
  }
  edsl::Tensor Reordered = edsl::Contraction().outShape(Reordered_dims).outAccess(Reordered_idxs).assign(I(I_idxs));

  std::vector<edsl::TensorDim> O_dims;
  O_dims.push_back(Reordered_dims[0]);
  O_dims.push_back(Reordered_dims[1]);
  for (size_t i = 0; i < I.rank() - 2; i++) {
    O_dims[1] = O_dims[1] * Reordered_dims[i + 2];
    O_dims.push_back(Reordered_dims[I.rank() + i]);
  }
  return edsl::make_tuple(op::reshape(Reordered, edsl::make_tuple(O_dims)));
});

}  // namespace PlaidMLPlugin
