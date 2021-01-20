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

void registerMatMul() {
  registerOp("MatMul", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset1::MatMul>(ctx.layer);
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

    std::vector<edsl::TensorDim> A_dims(ndimsA);
    std::vector<edsl::TensorDim> B_dims(ndimsB);
    std::vector<edsl::TensorDim> O_dims;
    std::vector<edsl::TensorIndex> A_idxs(ndimsA);
    std::vector<edsl::TensorIndex> B_idxs(ndimsB);
    std::vector<edsl::TensorIndex> O_idxs;

    edsl::TensorIndex z;
    A.bind_dims(A_dims);
    B.bind_dims(B_dims);

    A_idxs[ndimsA - 1] = z;
    B_idxs[ndimsB - 2] = z;
    for (size_t i = 0; i < ndimsA - 2; ++i) {
      A_idxs[i] = B_idxs[i];
    }

    for (size_t i = 0; i < ndimsA - 1; ++i) {
      O_dims.push_back(A_dims[i]);
      O_idxs.push_back(A_idxs[i]);
    }
    O_dims.push_back(B_dims[ndimsB - 1]);
    O_idxs.push_back(B_idxs[ndimsB - 1]);

    edsl::Tensor O = edsl::Contraction(O_dims, O_idxs).sum(A(A_idxs) * B(B_idxs));
    return edsl::make_tuple(O);
  });
}

}  // namespace PlaidMLPlugin
