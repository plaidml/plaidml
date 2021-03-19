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

// gemv_start
Tensor GEMV(const Tensor& A, const Tensor& x, const Tensor& y) {
  TensorDim I, J;
  TensorIndex i, j;
  A.bind_dims(I, J);
  x.bind_dims(J);
  return Contraction().outShape(I).outAccess(i).sum(A(i, j) * x(j)) + y;
}
// gemv_end

// constant_gemv_start
Tensor GEMV2(const Tensor& A, const Tensor& x, const Tensor& y, int alpha, int beta) {
  TensorDim I, J;
  TensorIndex i, j;
  x.bind_dims(J);
  Tensor A_alpha = A * alpha;
  A_alpha.bind_dims(I, J);
  Tensor y_beta = y * beta;
  return Contraction().outShape(I).outAccess(i).sum(A_alpha(i, j) * x(j)) + y_beta;
}
// constant_gemv_end

class ExampleCppEdsl : public TestFixture {};

TEST_F(ExampleCppEdsl, GEMVC_INT64) {
  auto A = Placeholder(DType::INT64, {3, 3});
  auto x = Placeholder(DType::INT64, {3});
  auto y = Placeholder(DType::INT64, {3});
  auto program = makeProgram("gemvc", {A, x, y}, {GEMV2(A, x, y, 5, 4)});

  std::vector<int64_t> data_A{1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int64_t> data_x{
      1,
      1,
      1,
  };
  std::vector<int64_t> data_y{1, 1, 1};
  std::vector<int64_t> data_O{19, 19, 19};
  checkExact(program, {data_A, data_x, data_y}, {data_O});
}

TEST_F(ExampleCppEdsl, GEMV_INT64) {
  auto A = Placeholder(DType::INT64, {3, 3});
  auto x = Placeholder(DType::INT64, {3});
  auto y = Placeholder(DType::INT64, {3});
  auto program = makeProgram("gemv", {A, x, y}, {GEMV(A, x, y)});

  std::vector<int64_t> data_A{1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int64_t> data_x{
      1,
      1,
      1,
  };
  std::vector<int64_t> data_y{1, 1, 1};
  std::vector<int64_t> data_O{4, 4, 4};
  checkExact(program, {data_A, data_x, data_y}, {data_O});
}

TEST_F(ExampleCppEdsl, GEMV_UINT64) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto x = Placeholder(DType::UINT64, {3});
  auto y = Placeholder(DType::UINT64, {3});
  auto program = makeProgram("gemv", {A, x, y}, {GEMV(A, x, y)});

  std::vector<uint64_t> data_A{1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> data_x{
      1,
      1,
      1,
  };
  std::vector<uint64_t> data_y{1, 1, 1};
  std::vector<uint64_t> data_O{4, 4, 4};
  checkExact(program, {data_A, data_x, data_y}, {data_O});
}

TEST_F(ExampleCppEdsl, GEMV_INT32) {
  auto A = Placeholder(DType::INT32, {3, 3});
  auto x = Placeholder(DType::INT32, {3});
  auto y = Placeholder(DType::INT32, {3});
  auto program = makeProgram("gemv", {A, x, y}, {GEMV(A, x, y)});

  std::vector<int32_t> data_A{1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> data_x{
      1,
      1,
      1,
  };
  std::vector<int32_t> data_y{1, 1, 1};
  std::vector<int32_t> data_O{4, 4, 4};
  checkExact(program, {data_A, data_x, data_y}, {data_O});
}

TEST_F(ExampleCppEdsl, GEMV_INT8) {
  auto A = Placeholder(DType::INT8, {3, 3});
  auto x = Placeholder(DType::INT8, {3});
  auto y = Placeholder(DType::INT8, {3});
  auto program = makeProgram("gemv", {A, x, y}, {GEMV(A, x, y)});

  std::vector<int8_t> data_A{1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int8_t> data_x{
      1,
      1,
      1,
  };
  std::vector<int8_t> data_y{1, 1, 1};
  std::vector<int8_t> data_O{4, 4, 4};
  checkExact(program, {data_A, data_x, data_y}, {data_O});
}

TEST_F(ExampleCppEdsl, GEMV_FLOAT32) {
  auto A = Placeholder(DType::FLOAT32, {3, 3});
  auto x = Placeholder(DType::FLOAT32, {3});
  auto y = Placeholder(DType::FLOAT32, {3});
  auto program = makeProgram("gemv", {A, x, y}, {GEMV(A, x, y)});

  std::vector<float> data_A{1, 0.5, 1, 1, 1, 1, 1, 0.6, 1};
  std::vector<float> data_x{
      1,
      1,
      1,
  };
  std::vector<float> data_y{1, 1, 1};
  std::vector<float> data_O{3.5, 4, 3.6};
  checkExact(program, {data_A, data_x, data_y}, {data_O});
}

}  // namespace plaidml::edsl
