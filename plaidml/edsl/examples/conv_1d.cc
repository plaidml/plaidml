// Copyright 2020 Intel Corporation.
// Note:
//    This file is being used by sphinx docs to pull in code blocks.
//    Code blocks are pulled into docs/usage/*.rst
//    Any changes made here may upset the docs.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "plaidml/edsl/edsl.h"
#include "plaidml/testenv.h"

namespace plaidml::edsl {

class ExampleCppEdsl : public TestFixture {};

// conv_1d_start
Tensor Conv1D(const Tensor& I, const Tensor& K) {
  TensorDim N, X, KX, CI, CO;
  TensorIndex n, x, k, ci, co;
  I.bind_dims(N, X, CI);
  K.bind_dims(KX, CI, CO);
  return Contraction().outShape(N, X - KX + 1, CO).outAccess(n, x, co).sum(I(n, x + k, ci) * K(k, ci, co));
}
// conv_1d_end

TEST_F(ExampleCppEdsl, Conv1D) {
  auto I = Placeholder(DType::UINT64, {1, 244, 3});
  auto K = Placeholder(DType::UINT64, {3, 3, 1});
  runProgram(makeProgram("conv_1d", {I, K}, {Conv1D(I, K)}));
}

}  // namespace plaidml::edsl
