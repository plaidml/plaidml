// Copyright 2020 Intel Corporation.
// Note:
//    This file is being used by sphinx docs to pull in code blocks.
//    Code blocks are pulled into docs/usage/*.rst
//    Any changes made here may upset the docs.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "plaidml/edsl/edsl.h"
#include "plaidml/testenv.h"

// The example below illustrates a maxpool operation on a one dimensional tensor.
// Pooling is a method commonly used in neural networks to downsize a tensor.
// Like the name implies, pooling is a type of contraction which groups elements of a tensor together,
// then performs an aggregation on them. In this particular case, we’ll be looking at a 1D Max Pool,
// which is a native operation in most popular frameworks:
// In Tensorflow this can be written as:
//       tf.keras.layers.MaxPool1D(pool_size=2)
// similarly in pytorch as:
//       torch.nn.MaxPool1d(kernel_size=2)
// Under the hood, this max pool splits a tensor into groups of 2 and takes the larger element from each group,
// yielding a tensor of half the original size. This is also quite straightforward to implement in C++/Python.
// See ForLoopMaxPool1D(...) below for reference.
// This for loop can be translated into eDSL as shown in MaxPool1D(...)

namespace plaidml::edsl {

// max_pool_1d_start
Tensor MaxPool1D(const Tensor& I) {
  TensorDim N;
  TensorIndex i, j;
  I.bind_dims(N);
  return Contraction().outShape(N / 2).outAccess(i).max(I(2 * i + j)).add_constraint(j < 2);
}
// max_pool_1d_end

// the code below is used in documentation for further explanation of the maxpool operation

void ForLoopMaxPool1D(float* I, float* O, int N) {
  // for_loop_max_pool_start
  for (int i = 0; i < N / 2; ++i) {
    float curr_max = std::numeric_limits<float>::min();
    for (int j = 0; j < 2; ++j) {
      if (I[2 * i + j] > curr_max) {
        curr_max = I[2 * i + j];
      }
    }
    O[i] = curr_max;
  }
  // for_loop_max_pool_end
}

// wrong_max_pool_start
Tensor WrongMaxPool1D(const Tensor& I) {
  TensorDim N;
  TensorIndex i, j;
  I.bind_dims(N);
  return Contraction().outShape(N / 2).outAccess(i).max(I(2 * i + j));
}
// wrong_max_pool_end

class ExampleCppEdsl : public TestFixture {};

TEST_F(ExampleCppEdsl, MaxPool1D) {
  auto I = Placeholder(DType::UINT64, {3});
  runProgram(makeProgram("max_pool_1d", {I}, {MaxPool1D(I)}));
}

TEST_F(ExampleCppEdsl, WrongMaxPool1D) {
  auto I = Placeholder(DType::UINT64, {3});
  makeProgram("wrong_max_pool_1d", {I}, {WrongMaxPool1D(I)});
}

}  // namespace plaidml::edsl
