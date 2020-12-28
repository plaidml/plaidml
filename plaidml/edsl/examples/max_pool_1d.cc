// Copyright 2020 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "plaidml/edsl/edsl.h"
#include "plaidml/testenv.h"

namespace plaidml::edsl {

class ExampleCppEdsl : public TestFixture {};

// max_pool_1d_start
Tensor MaxPool1D(const Tensor& I) {
  TensorDim N;
  TensorIndex i, j;
  I.bind_dims(N);
  return Contraction().outShape(N / 2).outAccess(i).max(I(2 * i + j)).add_constraint(j < 2);
}
// max_pool_1d_end

TEST_F(ExampleCppEdsl, MaxPool1D) {
  auto I = Placeholder(DType::UINT64, {3});
  runProgram(makeProgram("max_pool_1d", {I}, {MaxPool1D(I)}));
}

}  // namespace plaidml::edsl
