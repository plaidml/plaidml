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

// gemm_start
Tensor GEMM(const Tensor& A, const Tensor& B, const Tensor& C) {
  TensorDim I, J, K;
  TensorIndex i, j, k;
  A.bind_dims(I, K);
  B.bind_dims(K, J);
  return Contraction().outShape(I, J).outAccess(i, j).sum(A(i, k) * B(k, j)) + C;
}
// gemm_end

class ExampleCppEdsl : public TestFixture {};

TEST_F(ExampleCppEdsl, GEMM_INT64) {
  auto A = Placeholder(DType::INT64, {3, 3});
  auto B = Placeholder(DType::INT64, {3, 3});
  auto C = Placeholder(DType::INT64, {3, 3});
  auto program = makeProgram("gemm", {A, B, C}, {GEMM(A, B, C)});

  std::vector<int64_t> data_A{1, 1, 1, 1, 1, 2, 3, 3, 3};
  std::vector<int64_t> data_B{10, 11, 12, 13, 14, 15, 16, 17, 18};
  std::vector<int64_t> data_C{1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int64_t> data_O{40, 43, 46, 56, 60, 64, 118, 127, 136};
  checkExact(program, {data_A, data_B, data_C}, {data_O});
}

TEST_F(ExampleCppEdsl, GEMM_UINT64) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = Placeholder(DType::UINT64, {3, 3});
  auto C = Placeholder(DType::UINT64, {3, 3});
  auto program = makeProgram("gemm", {A, B, C}, {GEMM(A, B, C)});

  std::vector<uint64_t> data_A{1, 1, 1, 1, 1, 2, 3, 3, 3};
  std::vector<uint64_t> data_B{10, 11, 12, 13, 14, 15, 16, 17, 18};
  std::vector<uint64_t> data_C{1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> data_O{40, 43, 46, 56, 60, 64, 118, 127, 136};
  checkExact(program, {data_A, data_B, data_C}, {data_O});
}

TEST_F(ExampleCppEdsl, GEMM_INT32) {
  auto A = Placeholder(DType::INT32, {3, 3});
  auto B = Placeholder(DType::INT32, {3, 3});
  auto C = Placeholder(DType::INT32, {3, 3});
  auto program = makeProgram("gemm", {A, B, C}, {GEMM(A, B, C)});

  std::vector<int32_t> data_A{1, 1, 1, 1, 1, 2, 3, 3, 3};
  std::vector<int32_t> data_B{10, 11, 12, 13, 14, 15, 16, 17, 18};
  std::vector<int32_t> data_C{1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> data_O{40, 43, 46, 56, 60, 64, 118, 127, 136};
  checkExact(program, {data_A, data_B, data_C}, {data_O});
}

TEST_F(ExampleCppEdsl, GEMM_INT8) {
  auto A = Placeholder(DType::INT8, {3, 3});
  auto B = Placeholder(DType::INT8, {3, 3});
  auto C = Placeholder(DType::INT8, {3, 3});
  auto program = makeProgram("gemm", {A, B, C}, {GEMM(A, B, C)});

  std::vector<int8_t> data_A{1, 1, 1, 1, 1, 2, 3, 3, 3};
  std::vector<int8_t> data_B{1, 2, 3, 3, 4, 5, 3, 3, 3};
  std::vector<int8_t> data_C{1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int8_t> data_O{8, 10, 12, 11, 13, 15, 22, 28, 34};
  checkExact(program, {data_A, data_B, data_C}, {data_O});
}

TEST_F(ExampleCppEdsl, GEMM_FLOAT32) {
  auto A = Placeholder(DType::FLOAT32, {3, 3});
  auto B = Placeholder(DType::FLOAT32, {3, 3});
  auto C = Placeholder(DType::FLOAT32, {3, 3});
  auto program = makeProgram("gemm", {A, B, C}, {GEMM(A, B, C)});

  std::vector<float> data_A{0.5, 0.2, 4, 1, 1, 2, 3, 3, 0.3};
  std::vector<float> data_B{1, 2, 3, 3, 4, 0.5, 3, 3, 3};
  std::vector<float> data_C{1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<float> data_O{14.1, 14.8, 14.6, 11, 13, 10.5, 13.9, 19.9, 12.4};
  checkExact(program, {data_A, data_B, data_C}, {data_O});
}

}  // namespace plaidml::edsl
