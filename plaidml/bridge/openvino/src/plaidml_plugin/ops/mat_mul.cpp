// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("matmul", [](const Context& ctx) {
  auto* layer = dynamic_cast<ngraph::opset1::MatMul*>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 2);
  auto A = ctx.operands.at(0);
  auto B = ctx.operands.at(1);

  auto ndimsA = A.rank();
  auto ndimsB = B.rank();

  if (ndimsA < 2) {
    A = op::unsqueeze(A, {0});
    ndimsA++;
  }
  if (ndimsB < 2) {
    B = op::unsqueeze(B, {0});
    ndimsB++;
  }
  if (layer->get_transpose_a()) {
    std::vector<int64_t> pattern;
    for (size_t i = 0; i < ndimsA - 2; i++) {
      pattern.push_back(i);
    }
    pattern.push_back(ndimsA - 1);
    pattern.push_back(ndimsA - 2);
    A = op::transpose(A, edsl::make_tuple(pattern));
  }
  if (layer->get_transpose_b()) {
    std::vector<int64_t> pattern;
    for (size_t i = 0; i < ndimsB - 2; i++) {
      pattern.push_back(i);
    }
    pattern.push_back(ndimsB - 1);
    pattern.push_back(ndimsB - 2);
    B = op::transpose(B, edsl::make_tuple(pattern));
  }
  while (A.rank() < B.rank()) {
    A = op::unsqueeze(A, {0});
  }
  while (B.rank() < A.rank()) {
    B = op::unsqueeze(B, {0});
  }

  return edsl::make_tuple(op::dot(A, B));
});

}  // namespace PlaidMLPlugin
