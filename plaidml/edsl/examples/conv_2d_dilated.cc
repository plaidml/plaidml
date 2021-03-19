// Copyright 2021 Intel Corporation.
// Note:
//    This file is being used by sphinx docs to pull in code blocks.
//    Code blocks are pulled into docs/usage/*.rst
//    Any changes made here may upset the docs.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "plaidml/edsl/edsl.h"
#include "plaidml/testenv.h"

namespace plaidml::edsl {

// conv_2d_dilated_start
Tensor Conv2DDilated(const Tensor& I, const Tensor& K) {
  TensorDim N, X, Y, KX, KY, CI, CO;
  TensorIndex n, x, y, kx, ky, ci, co;
  I.bind_dims(N, X, Y, CI);
  K.bind_dims(KX, KY, CI, CO);
  return Contraction()
      .outShape(N, X - 2 * (KX - 1), Y - 3 * (KY - 1), CO)
      .outAccess(n, x, y, co)
      .sum(I(n, x + 2 * kx, y + 3 * ky, ci) * K(kx, ky, ci, co));
}
// conv_2d_dilated_end

class ExampleCppEdsl : public TestFixture {};

TEST_F(ExampleCppEdsl, Conv2DDilated) {
  auto I = Placeholder(DType::UINT64, {1, 244, 244, 1});
  auto K = Placeholder(DType::UINT64, {3, 3, 1, 32});
  runProgram(makeProgram("conv_2d_dilated", {I, K}, {Conv2DDilated(I, K)}));
}

}  // namespace plaidml::edsl
