// Copyright 2020 Intel Corporation
//
// N.B. When running via lit, we always use the llvm_cpu device.
// RUN: cc_test \
// RUN:   --plaidml_device=llvm_cpu.0 \
// RUN:   --plaidml_target=llvm_cpu \
// RUN:   --generate_filecheck_input \
// RUN: | FileCheck %s

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>

#include "half.hpp"
#include "llvm/ADT/StringRef.h"

#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"
#include "plaidml/testenv.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

using half_float::half;
using llvm::StringRef;
using ::testing::ContainerEq;
using ::testing::Eq;

namespace plaidml::edsl {

namespace {

class CppEdsl : public TestFixture {};

template <typename T>
Buffer makeBuffer(DType dtype, const std::vector<int64_t>& dims, const std::vector<T>& data) {
  TensorShape shape(dtype, dims);
  Buffer buffer(shape);
  buffer.copy_from(data.data());
  return buffer;
}

Tensor Dot(Tensor X, Tensor Y) {
  TensorDim I, J, K;
  TensorIndex i, j, k;
  X.bind_dims(I, K);
  Y.bind_dims(K, J);
  return Contraction().outShape(I, J).outAccess(i, j).sum(X(i, k) * Y(k, j));
}

Tensor Relu(Tensor I) {
  auto zero = cast(Tensor(0.0), I.dtype());
  return select(I < 0.0, zero, I);
}

Tensor Softmax(Tensor X) {
  TensorDim I, J;
  TensorIndex i, j;
  X.bind_dims(I, J);
  Tensor M = Contraction().outShape(I, 1).outAccess(i, 0).max(X(i, j));
  auto E = exp(X - M);
  Tensor N = Contraction().outShape(I, 1).outAccess(i, 0).sum(E(i, j));
  return E / N;
}

TEST_F(CppEdsl, BindDims) {
  const int64_t M = 8;
  const int64_t N = 32;
  const int64_t K = 16;
  auto A = Placeholder(DType::FLOAT32, {M, K});
  auto B = Placeholder(DType::FLOAT32, {K, N});
  auto C = Placeholder(DType::FLOAT32, {K, 0});

  EXPECT_NO_THROW({ A.bind_dims(M, K); });
  EXPECT_NO_THROW({ C.bind_dims(K, K); });
  EXPECT_ANY_THROW({ A.bind_dims(0, 0); });

  {
    TensorDim D0, D1, D2;
    EXPECT_NO_THROW({
      A.bind_dims(D0, D1);
      B.bind_dims(D1, D2);
    });
  }

  {
    TensorDim D0, D1, D2;
    EXPECT_ANY_THROW({
      A.bind_dims(D0, D1);
      B.bind_dims(D0, D2);
    });
  }

  {
    TensorDim D0, D1, D2;
    EXPECT_NO_THROW({
      A.bind_dims(D0, D1);
      C.bind_dims(D1, D2);
    });
  }
}

TEST_F(CppEdsl, HigherPrecisionConstants) {
  auto A = Placeholder(DType::FLOAT32, {3, 3});
  auto C = A + cast(Tensor{1}, DType::UINT64) + cast(Tensor{2.0}, DType::FLOAT64);
  auto program = makeProgram("higher_precision_constants", {A}, {C});

  // clang-format off
  // CHECK-LABEL: CppEdsl.HigherPrecisionConstants
  // CHECK: module @higher_precision_constants
  // CHECK: tile.add %{{.*}}, %{{.*}} : (tensor<3x3xf32>, tensor<ui64>) -> tensor<3x3xf32>
  // CHECK: tile.add %{{.*}}, %{{.*}} : (tensor<3x3xf32>, tensor<f64>) -> tensor<3x3xf64>
  // CHECK: return %{{.*}} : tensor<3x3xf64>
  // clang-format on

  std::vector<float> A_input{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> C_output{4, 5, 6, 7, 8, 9, 10, 11, 12};
  checkExact(program, {A_input}, {C_output});
}

TEST_F(CppEdsl, Cast) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = cast(A, DType::UINT32);
  auto program = makeProgram("cast", {A}, {B});

  std::vector<uint64_t> A_input{1,
                                2,
                                3,
                                4,
                                5,
                                6 + (1UL << 12),
                                7 + (1UL << 24),
                                8 + (1UL << 31),  //
                                (1ULL << 32) - 1};
  std::vector<uint32_t> B_output{1,
                                 2,
                                 3,
                                 4,
                                 5,
                                 6 + (1UL << 12),
                                 7 + (1UL << 24),
                                 8 + (1UL << 31),  //
                                 (1ULL << 32) - 1};
  checkExact(program, {A_input}, {B_output});
}

TEST_F(CppEdsl, BitAndScalar) {
  const uint64_t ONE = 1;

  auto A = Placeholder(DType::UINT64, {3, 3});
  uint64_t mask = UINT32_MAX;
  auto B = A & mask;
  auto program = makeProgram("bit_and", {A}, {B});

  std::vector<uint64_t> A_input{(ONE << 32),     (ONE << 33) + 1, (ONE << 34) + 2,  //
                                (ONE << 35) + 3, (ONE << 36) + 4, (ONE << 37) + 5,  //
                                (ONE << 38) + 6, (ONE << 39) + 7, (ONE << 40) + 8};
  std::vector<uint64_t> B_output{0, 1, 2,  //
                                 3, 4, 5,  //
                                 6, 7, 8};
  checkExact(program, {A_input}, {B_output});
}

TEST_F(CppEdsl, BitAnd) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = Placeholder(DType::UINT64, {3, 3});
  auto C = A & B;
  auto program = makeProgram("bit_and", {A, B}, {C});

  std::vector<uint64_t> A_input{1, 2, 3,  //
                                4, 5, 6,  //
                                7, 8, 9};
  std::vector<uint64_t> B_input{10, 11, 12,  //
                                13, 14, 15,  //
                                16, 17, 18};
  std::vector<uint64_t> C_output{1 & 10, 2 & 11, 3 & 12,  //
                                 4 & 13, 5 & 14, 6 & 15,  //
                                 7 & 16, 8 & 17, 9 & 18};
  checkExact(program, {A_input, B_input}, {C_output});
}

TEST_F(CppEdsl, BitOr) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = Placeholder(DType::UINT64, {3, 3});
  auto C = A | B;
  auto program = makeProgram("bit_or", {A, B}, {C});

  std::vector<uint64_t> A_input{1, 2, 3,  //
                                4, 5, 6,  //
                                7, 8, 9};
  std::vector<uint64_t> B_input{10, 11, 12,  //
                                13, 14, 15,  //
                                16, 17, 18};
  std::vector<uint64_t> C_output{1 | 10, 2 | 11, 3 | 12,  //
                                 4 | 13, 5 | 14, 6 | 15,  //
                                 7 | 16, 8 | 17, 9 | 18};
  checkExact(program, {A_input, B_input}, {C_output});
}

TEST_F(CppEdsl, BitLeft) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = Placeholder(DType::UINT64, {3, 3});
  auto C = A << B;
  auto program = makeProgram("bit_left", {A, B}, {C});

  std::vector<uint64_t> A_input{1, 2, 3,  //
                                4, 5, 6,  //
                                7, 8, 9};
  std::vector<uint64_t> B_input{10, 11, 12,  //
                                13, 14, 15,  //
                                16, 17, 18};
  std::vector<uint64_t> C_output{1 << 10, 2 << 11, 3 << 12,  //
                                 4 << 13, 5 << 14, 6 << 15,  //
                                 7 << 16, 8 << 17, 9 << 18};
  checkExact(program, {A_input, B_input}, {C_output});
}

TEST_F(CppEdsl, BitRightTensor) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = Placeholder(DType::UINT64, {3, 3});
  auto C = A >> B;
  auto program = makeProgram("bit_right_tensor", {A, B}, {C});

  std::vector<uint64_t> A_input{1 << 10, 2 << 11, 3 << 12,  //
                                4 << 13, 5 << 14, 6 << 15,  //
                                7 << 16, 8 << 17, 9 << 18};
  std::vector<uint64_t> B_input{10, 11, 12,  //
                                13, 14, 15,  //
                                16, 17, 18};
  std::vector<uint64_t> C_output{1, 2, 3,  //
                                 4, 5, 6,  //
                                 7, 8, 9};
  checkExact(program, {A_input, B_input}, {C_output});
}

TEST_F(CppEdsl, BitRightScalar) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = A >> 9;
  auto program = makeProgram("bit_right_scalar", {A}, {B});

  std::vector<uint64_t> A_input{1 << 10, 2 << 11, 3 << 12,  //
                                4 << 13, 5 << 14, 6 << 15,  //
                                7 << 16, 8 << 17, 9 << 18};
  std::vector<uint64_t> B_output{1 << 1, 2 << 2, 3 << 3,  //
                                 4 << 4, 5 << 5, 6 << 6,  //
                                 7 << 7, 8 << 8, 9 << 9};
  checkExact(program, {A_input}, {B_output});
}

TEST_F(CppEdsl, BitNot) {
  auto A = Placeholder(DType::UINT8, {3, 3});
  auto B = ~A;
  auto program = makeProgram("bit_not", {A}, {B});

  std::vector<uint8_t> A_input{0x00, 0x01, 0x02,  //
                               0x10, 0x11, 0x22,  //
                               0xF0, 0x0F, 0xFF};
  std::vector<uint8_t> B_output{0xFF, 0xFE, 0xFD,  //
                                0xEF, 0xEE, 0xDD,  //
                                0x0F, 0xF0, 0x00};
  checkExact(program, {A_input}, {B_output});
}

TEST_F(CppEdsl, BitXor) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = Placeholder(DType::UINT64, {3, 3});
  auto C = A ^ B;
  auto program = makeProgram("bit_xor", {A, B}, {C});

  std::vector<uint64_t> A_input{1, 2, 3,  //
                                4, 5, 6,  //
                                7, 8, 9};
  std::vector<uint64_t> B_input{10, 11, 12,  //
                                13, 14, 15,  //
                                16, 17, 18};
  std::vector<uint64_t> C_output{0x1 ^ 10, 0x2 ^ 11, 0x3 ^ 12,  //
                                 0x4 ^ 13, 0x5 ^ 14, 0x6 ^ 15,  //
                                 0x7 ^ 16, 0x8 ^ 17, 0x9 ^ 18};
  checkExact(program, {A_input, B_input}, {C_output});
}

TEST_F(CppEdsl, BroadcastCmp) {
  auto A = Placeholder(DType::UINT64, {3, 4});
  auto B = Placeholder(DType::UINT64, {3, 1});
  auto C = cast(A >= B, DType::UINT64);
  auto program = makeProgram("broadcast_cmp", {A, B}, {C});

  std::vector<uint64_t> A_input = {0, 1, 2,  3,  //
                                   4, 5, 6,  7,  //
                                   8, 9, 10, 11};
  std::vector<uint64_t> B_input = {0, 6, 12};
  std::vector<uint64_t> C_output = {1, 1, 1, 1,  //
                                    0, 0, 1, 1,  //
                                    0, 0, 0, 0};
  checkExact(program, {A_input, B_input}, {C_output});
}

TEST_F(CppEdsl, Add) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = Placeholder(DType::UINT64, {3, 3});
  auto C = A + B;
  auto program = makeProgram("add", {A, B}, {C});

  std::vector<uint64_t> A_input = {
      1,
      2,
      3,
      4,
      5,
      6 + (1UL << 12),
      7 + (1UL << 24),
      8 + (1ULL << 32),
      9 + (1ULL << 40)  //
  };

  std::vector<uint64_t> B_input = {1,
                                   2 + (1UL << 12),
                                   3,
                                   4 + (1UL << 24),
                                   5,
                                   6 + (1ULL << 32),
                                   7,
                                   8 + (1ULL << 40),  //
                                   9};

  std::vector<uint64_t> C_output = {2,
                                    4 + (1UL << 12),
                                    6,
                                    8 + (1UL << 24),
                                    10,
                                    12 + (1UL << 12) + (1ULL << 32),
                                    14 + (1UL << 24),
                                    16 + (1ULL << 32) + (1ULL << 40),
                                    18 + (1ULL << 40)};

  checkExact(program, {A_input, B_input}, {C_output});
}

TEST_F(CppEdsl, ConstAdd) {
  std::vector<int64_t> shape = {4};
  std::vector<int> a = {4, 3, 2, 1};
  std::vector<int> b = {1, 2, 3, 4};
  auto A = Constant(makeBuffer(DType::INT32, shape, a), "A");
  auto B = Constant(makeBuffer(DType::INT32, shape, b), "B");
  auto O = A + B;
  auto program = makeProgram("const_add", {}, {O});

  // clang-format off
  // CHECK-LABEL: CppEdsl.ConstAdd
  // CHECK: module @const_add
  // CHECK: tile.add %{{.*}}, %{{.*}} : (tensor<4xsi32>, tensor<4xsi32>) -> tensor<4xsi32>
  // CHECK: return %{{.*}} : tensor<4xsi32>
  // clang-format on

  std::vector<int32_t> expected = {5, 5, 5, 5};
  checkExact(program, {}, {expected});
}

TEST_F(CppEdsl, ConstCast) {
  auto O = cast(Tensor{3}, DType::FLOAT32);
  auto program = makeProgram("const_cast", {}, {O});
  std::vector<float> expected = {3.0};
  checkExact(program, {}, {expected});
}

TEST_F(CppEdsl, Dot) {
  const int64_t M = 8;
  const int64_t N = 32;
  const int64_t K = 16;
  auto A = Placeholder(DType::FLOAT32, {M, K});
  auto B = Placeholder(DType::FLOAT32, {K, N});
  auto C = Dot(A, B);
  auto program = makeProgram("dot", {A, B}, {C});

  // clang-format off
  // CHECK-LABEL: CppEdsl.Dot
  // CHECK: module @dot
  // CHECK: %[[cst:.*]] = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK: %[[cion:.*]] = tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {{{.*}}}
  // CHECK-SAME: tensor<f32>, tensor<8x16xf32>, tensor<16x32xf32> -> tensor<8x32xf32>
  // CHECK: return %[[cion]] : tensor<8x32xf32>
  // clang-format on

  std::default_random_engine rng(2);
  std::normal_distribution<float> normal_dist(0.0, 1.0);

  std::vector<float> in1(M * K);
  for (unsigned i = 0; i < in1.size(); i++) {
    in1[i] = normal_dist(rng);
  }
  std::vector<float> in2(K * N);
  for (unsigned i = 0; i < in2.size(); i++) {
    in2[i] = normal_dist(rng);
  }
  std::vector<float> expected(M * N);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        expected[i * N + j] += in1[i * K + k] * in2[k * N + j];
      }
    }
  }
  checkClose(program, {in1, in2}, {expected});
}

TEST_F(CppEdsl, DotF16) {
  const int64_t M = 8;
  const int64_t N = 32;
  const int64_t K = 16;
  Tensor A = Placeholder(DType::FLOAT16, {M, K});
  Tensor B = Placeholder(DType::FLOAT16, {K, N});
  auto C = Dot(A, B);
  auto program = makeProgram("dot_f16", {A, B}, {C});

  // clang-format off
  // CHECK-LABEL: CppEdsl.DotF16
  // CHECK: module @dot_f16
  // CHECK: %[[cst:.*]] = tile.constant(0.000000e+00 : f64) : tensor<f16>
  // CHECK: %[[cion:.*]] = tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {{{.*}}}
  // CHECK-SAME: tensor<f16>, tensor<8x16xf16>, tensor<16x32xf16> -> tensor<8x32xf16>
  // CHECK: return %[[cion]] : tensor<8x32xf16>
  // clang-format on
}

TEST_F(CppEdsl, DotF16_AccF32) {
  const int64_t M = 8;
  const int64_t N = 32;
  const int64_t K = 16;
  Tensor A = Placeholder(DType::FLOAT16, {M, K});
  Tensor B = Placeholder(DType::FLOAT16, {K, N});

  Tensor A_f32 = cast(A, DType::FLOAT32);
  Tensor B_f32 = cast(B, DType::FLOAT32);
  TensorIndex i, j, k;
  Tensor C_f32 = Contraction().outShape(M, N).outAccess(i, j).sum(A_f32(i, k) * B_f32(k, j));
  Tensor C = cast(C_f32, DType::FLOAT16);

  auto program = makeProgram("dot_f16_acc_f32", {A, B}, {C});

  std::default_random_engine rng(2);
  std::normal_distribution<float> normal_dist(0.0, 1.0);

  std::vector<half> in1(M * K);
  for (unsigned i = 0; i < in1.size(); i++) {
    in1[i] = normal_dist(rng);
  }
  std::vector<half> in2(K * N);
  for (unsigned i = 0; i < in2.size(); i++) {
    in2[i] = normal_dist(rng);
  }
  std::vector<float> acc(M * N);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        acc[i * N + j] += in1[i * K + k] * in2[k * N + j];
      }
    }
  }
  std::vector<half> expected(acc.begin(), acc.end());
  checkClose(program, {in1, in2}, {expected}, /*tolerance=*/1e-2);
}

TEST_F(CppEdsl, DoubleDot) {
  auto A = Placeholder(DType::FLOAT32, {10, 20});
  auto B = Placeholder(DType::FLOAT32, {20, 30});
  auto C = Placeholder(DType::FLOAT32, {30, 40});
  auto program = makeProgram("double_dot", {A, B, C}, {Dot(Dot(A, B), C)});

  // clang-format off
  // CHECK-LABEL: CppEdsl.DoubleDot
  // CHECK: module @double_dot
  // CHECK: -> tensor<10x40xf32> {
  // CHECK: %[[cst:.*]] = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK: tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}, #map{{[0-9]+}}]} : tensor<f32>, tensor<10x20xf32>, tensor<20x30xf32> -> tensor<10x30xf32>
  // CHECK: tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}, #map{{[0-9]+}}]} : tensor<f32>, tensor<10x30xf32>, tensor<30x40xf32> -> tensor<10x40xf32>
  // CHECK: return %{{.*}} : tensor<10x40xf32>
  // clang-format on
  runProgram(program);
}

TEST_F(CppEdsl, BigDot) {
  int64_t M = 2048;
  int64_t N = 2048;
  int64_t K = 2048;
  auto A = Placeholder(DType::FLOAT32, {M, K});
  auto B = Placeholder(DType::FLOAT32, {K, N});
  auto C = Dot(A, B);
  auto program = makeProgram("dot", {A, B}, {C});
  runProgram(program);
}

TEST_F(CppEdsl, Max) {
  auto A = Placeholder(DType::FLOAT32, {3, 3});
  TensorDim I, J, K;
  TensorIndex i("i"), j("j");
  A.bind_dims(I, K);
  Tensor R = Contraction().outShape(I).outAccess(i).max(A(i, j));
  auto program = makeProgram("max", {A}, {R});
  std::vector<float> input = {
      -5.0f, -6.0f, -7.0f,  //
      4.0f,  5.0f,  6.0f,   //
      7.0f,  8.0f,  9.0f,   //
  };
  std::vector<float> expected = {-5.0, 6.0, 9.0};
  checkExact(program, {input}, {expected});
}

TEST_F(CppEdsl, EltwiseAdd) {
  auto A = Placeholder(DType::FLOAT32, {10, 20});
  auto B = Placeholder(DType::FLOAT32, {10, 20});
  auto program = makeProgram("eltwise_add", {A, B}, {A + B});

  // clang-format off
  // CHECK-LABEL: CppEdsl.EltwiseAdd
  // CHECK: module @eltwise_add
  // CHECK: tile.add %{{.*}}, %{{.*}} : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  // CHECK: return %{{.*}} : tensor<10x20xf32>
  // clang-format on
  runProgram(program);
}

TEST_F(CppEdsl, EltwiseMod) {
  auto A = Placeholder(DType::INT32, {3, 3});
  auto B = Placeholder(DType::INT32, {3, 3});
  auto C = A % B;
  auto program = makeProgram("mod", {A, B}, {C});

  // clang-format off
  // CHECK-LABEL: CppEdsl.EltwiseMod
  // CHECK: module @mod
  // CHECK: tile.mod %{{.*}}, %{{.*}} : (tensor<3x3xsi32>, tensor<3x3xsi32>) -> tensor<3x3xsi32>
  // CHECK: return %{{.*}} : tensor<3x3xsi32>
  // clang-format on

  std::vector<int32_t> A_input{2,   4,   8,   //
                               16,  32,  64,  //
                               128, 256, 512};
  std::vector<int32_t> B_input{1, 2, 3,  //
                               4, 5, 6,  //
                               7, 8, 9};
  std::vector<int32_t> C_output{2 % 1,   4 % 2,   8 % 3,   //
                                16 % 4,  32 % 5,  64 % 6,  //
                                128 % 7, 256 % 8, 512 % 9};
  checkExact(program, {A_input, B_input}, {C_output});
}

TEST_F(CppEdsl, Relu) {
  auto A = Placeholder(DType::FLOAT32, {10, 20});
  auto program = makeProgram("relu", {A}, {Relu(A)});
  // clang-format off
  // CHECK-LABEL: CppEdsl.Relu
  // CHECK: module @relu
  // CHECK: %[[cst:.*]] = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK: tile.cmp_lt %{{.*}}, %[[cst]] : (tensor<10x20xf32>, tensor<f32>) -> tensor<10x20xi1>
  // CHECK: tile.select %{{.*}}, %[[cst]], %{{.*}} : (tensor<10x20xi1>, tensor<f32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  // CHECK: return %{{.*}} : tensor<10x20xf32>
  // clang-format on
  runProgram(program);
}

TEST_F(CppEdsl, MnistMlp) {
  // model.add(Dense(512, activation='relu', input_shape=(784,)))
  auto input = Placeholder(DType::FLOAT32, {1, 784});
  auto kernel1 = Placeholder(DType::FLOAT32, {784, 512});
  auto bias1 = Placeholder(DType::FLOAT32, {512});
  auto dense1 = Relu(Dot(input, kernel1) + bias1);
  // model.add(Dense(512, activation='relu'))
  auto kernel2 = Placeholder(DType::FLOAT32, {512, 512});
  auto bias2 = Placeholder(DType::FLOAT32, {512});
  auto dense2 = Relu(Dot(dense1, kernel2) + bias2);
  // model.add(Dense(10, activation='softmax'))
  auto kernel3 = Placeholder(DType::FLOAT32, {512, 10});
  auto bias3 = Placeholder(DType::FLOAT32, {10});
  auto dense3 = Softmax(Dot(dense2, kernel3) + bias3);
  auto program = makeProgram("mnist_mlp", {input, kernel1, bias1, kernel2, bias2, kernel3, bias3}, {dense3});
  // clang-format off
  // CHECK-LABEL: CppEdsl.MnistMlp
  // CHECK: module @mnist_mlp
  // CHECK-DAG: %[[cst:.*]] = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK-DAG: %[[cst0:.*]] = tile.constant(0xFFF0000000000000 : f64) : tensor<f32>
  // CHECK: %[[X0:.*]] = tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}, #map{{[0-9]+}}]} : tensor<f32>, tensor<1x784xf32>, tensor<784x512xf32> -> tensor<1x512xf32>
  // CHECK: %[[X1:.*]] = tile.add %{{.*}}, %{{.*}} : (tensor<1x512xf32>, tensor<512xf32>) -> tensor<1x512xf32>
  // CHECK: %[[X2:.*]] = tile.cmp_lt %{{.*}}, %[[cst]] : (tensor<1x512xf32>, tensor<f32>) -> tensor<1x512xi1>
  // CHECK: %[[X3:.*]] = tile.select %{{.*}}, %[[cst]], %{{.*}} : (tensor<1x512xi1>, tensor<f32>, tensor<1x512xf32>) -> tensor<1x512xf32>
  // CHECK: %[[X4:.*]] = tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}, #map{{[0-9]+}}]} : tensor<f32>, tensor<1x512xf32>, tensor<512x512xf32> -> tensor<1x512xf32>
  // CHECK: %[[X5:.*]] = tile.add %{{.*}}, %{{.*}} : (tensor<1x512xf32>, tensor<512xf32>) -> tensor<1x512xf32>
  // CHECK: %[[X6:.*]] = tile.cmp_lt %{{.*}}, %[[cst]] : (tensor<1x512xf32>, tensor<f32>) -> tensor<1x512xi1>
  // CHECK: %[[X7:.*]] = tile.select %{{.*}}, %[[cst]], %{{.*}} : (tensor<1x512xi1>, tensor<f32>, tensor<1x512xf32>) -> tensor<1x512xf32>
  // CHECK: %[[X8:.*]] = tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}, #map{{[0-9]+}}]} : tensor<f32>, tensor<1x512xf32>, tensor<512x10xf32> -> tensor<1x10xf32>
  // CHECK: %[[X9:.*]] = tile.add %{{.*}}, %{{.*}} : (tensor<1x10xf32>, tensor<10xf32>) -> tensor<1x10xf32>
  // CHECK: %[[X10:.*]] = tile.contract max, none, %[[cst0]], %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}]} : tensor<f32>, tensor<1x10xf32> -> tensor<1x1xf32>
  // CHECK: %[[X11:.*]] = tile.sub %{{.*}}, %{{.*}} : (tensor<1x10xf32>, tensor<1x1xf32>) -> tensor<1x10xf32>
  // CHECK: %[[X12:.*]] = tile.exp %{{.*}} : (tensor<1x10xf32>) -> tensor<1x10xf32>
  // CHECK: %[[X13:.*]] = tile.contract add, none, %[[cst]], %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}]} : tensor<f32>, tensor<1x10xf32> -> tensor<1x1xf32>
  // CHECK: %[[X14:.*]] = tile.div %{{.*}}, %{{.*}} : (tensor<1x10xf32>, tensor<1x1xf32>) -> tensor<1x10xf32>
  // CHECK: return %{{.*}} : tensor<1x10xf32>
  // clang-format on
  runProgram(program);
}

Tensor Convolution2(Tensor I, Tensor K, const std::string& I_layout = "NHWC", const std::string& K_layout = "HWCK") {
  TensorLens I_lens(I_layout, "NHWC");
  TensorLens K_lens(K_layout, "HWCK");
  I = I.use(I_lens);
  K = K.use(K_lens);
  TensorDim CI, CO, K0, K1, N, X0, X1;
  TensorIndex n, x0, x1, co, ci, k0, k1;
  I.bind_dims(N, X0, X1, CI);
  K.bind_dims(K0, K1, CI, CO);
  return Contraction(I_lens)
      .outShape(N, X0, X1, CO)
      .outAccess(n, x0, x1, co)
      .sum(I(n, x0 + k0 - (K0 / 2), x1 + k1 - (K1 / 2), ci) * K(k0, k1, ci, co));
}

TEST_F(CppEdsl, Convolution) {
  auto I = Placeholder(DType::FLOAT32, {1, 56, 56, 64});
  auto K = Placeholder(DType::FLOAT32, {3, 3, 64, 64});
  auto program = makeProgram("convolution", {I, K}, {Convolution2(I, K)});
  // clang-format off
  // CHECK-LABEL: CppEdsl.Convolution
  // CHECK: module @convolution
  // CHECK: %[[cst:.*]] = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK: tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}, #map{{[0-9]+}}]} : tensor<f32>, tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32> -> tensor<1x56x56x64xf32>
  // CHECK: return %{{.*}} : tensor<1x56x56x64xf32>
  // clang-format on
  runProgram(program);
}

Tensor MaxPooling2(Tensor I) {
  TensorDim N, X0, X1, C;
  TensorIndex n, x0, x1, i, j, c;
  I.bind_dims(N, X0, X1, C);
  return Contraction()
      .outShape(N, (X0 + 1) / 2, (X1 + 1) / 2, C)
      .outAccess(n, x0, x1, c)
      .max(I(n, 2 * x0 + i, 2 * x1 + j, c))
      .add_constraint(i < 2)
      .add_constraint(j < 2);
}

Tensor Flatten(Tensor X) {
  std::vector<TensorDim> X_dims(X.rank());
  X.bind_dims(X_dims);
  if (X_dims.empty()) {
    return X;
  }
  TensorDim product{1};
  for (size_t i = 1; i < X_dims.size() - 1; i++) {
    product = product * X_dims[i];
  }
  return reshape(X, {TensorDim{1}, product});
}

TEST_F(CppEdsl, MnistCnn) {
  // model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
  auto input = Placeholder(DType::FLOAT32, {1, 224, 224, 1});
  auto kernel1 = Placeholder(DType::FLOAT32, {3, 3, 1, 32});
  auto bias1 = Placeholder(DType::FLOAT32, {32});
  auto conv1 = Relu(Convolution2(input, kernel1) + bias1);
  // model.add(Conv2D(64, (3, 3), activation='relu'))
  auto kernel2 = Placeholder(DType::FLOAT32, {3, 3, 32, 64});
  auto bias2 = Placeholder(DType::FLOAT32, {64});
  auto conv2 = Relu(Convolution2(conv1, kernel2) + bias2);
  // model.add(MaxPooling2D(pool_size=(2, 2)))
  auto pool1 = MaxPooling2(conv2);
  // model.add(Flatten())
  auto flat = Flatten(pool1);
  EXPECT_THAT(flat.compute_shape(), Eq(TensorShape(DType::FLOAT32, {1, 12544})));
  // model.add(Dense(128, activation='relu'))
  auto kernel3 = Placeholder(DType::FLOAT32, {12544, 128});
  auto bias3 = Placeholder(DType::FLOAT32, {128});
  auto dense1 = Relu(Dot(flat, kernel3) + bias3);
  const int64_t kNumClasses = 100;
  // model.add(Dense(num_classes, activation='softmax'))
  auto kernel4 = Placeholder(DType::FLOAT32, {128, kNumClasses});
  auto bias4 = Placeholder(DType::FLOAT32, {kNumClasses});
  auto dense2 = Softmax(Dot(dense1, kernel4) + bias4);
  auto program =
      makeProgram("mnist_cnn", {input, kernel1, bias1, kernel2, bias2, kernel3, bias3, kernel4, bias4}, {dense2});
  // clang-format off
  // CHECK-LABEL: CppEdsl.MnistCnn
  // CHECK: module @mnist_cnn
  // CHECK-DAG: %[[cst:.*]] = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK-DAG: %[[cst_0:.*]] = tile.constant(0xFFF0000000000000 : f64) : tensor<f32>
  // CHECK: tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}, #map{{[0-9]+}}]} : tensor<f32>, tensor<1x224x224x1xf32>, tensor<3x3x1x32xf32> -> tensor<1x224x224x32xf32>
  // CHECK: tile.add %{{.*}}, %{{.*}} : (tensor<1x224x224x32xf32>, tensor<32xf32>) -> tensor<1x224x224x32xf32>
  // CHECK: tile.cmp_lt %{{.*}}, %[[cst]] : (tensor<1x224x224x32xf32>, tensor<f32>) -> tensor<1x224x224x32xi1>
  // CHECK: tile.select %{{.*}}, %[[cst]], %{{.*}} : (tensor<1x224x224x32xi1>, tensor<f32>, tensor<1x224x224x32xf32>) -> tensor<1x224x224x32xf32>
  // CHECK: tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}, #map{{[0-9]+}}]} : tensor<f32>, tensor<1x224x224x32xf32>, tensor<3x3x32x64xf32> -> tensor<1x224x224x64xf32>
  // CHECK: tile.add %{{.*}}, %{{.*}} : (tensor<1x224x224x64xf32>, tensor<64xf32>) -> tensor<1x224x224x64xf32>
  // CHECK: tile.cmp_lt %{{.*}}, %[[cst]] : (tensor<1x224x224x64xf32>, tensor<f32>) -> tensor<1x224x224x64xi1>
  // CHECK: tile.select %{{.*}}, %[[cst]], %{{.*}} : (tensor<1x224x224x64xi1>, tensor<f32>, tensor<1x224x224x64xf32>) -> tensor<1x224x224x64xf32>
  // CHECK: tile.contract max, none, %[[cst_0]], %{{.*}} {cons = #set{{[0-9]+}}, sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}]} : tensor<f32>, tensor<1x224x224x64xf32> -> tensor<1x112x112x64xf32>
  // CHECK: tile.reshape %{{.*}} : (tensor<1x112x112x64xf32>) -> tensor<1x12544xf32>
  // CHECK: tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}, #map{{[0-9]+}}]} : tensor<f32>, tensor<1x12544xf32>, tensor<12544x128xf32> -> tensor<1x128xf32>
  // CHECK: tile.add %{{.*}}, %{{.*}} : (tensor<1x128xf32>, tensor<128xf32>) -> tensor<1x128xf32>
  // CHECK: tile.cmp_lt %{{.*}}, %[[cst]] : (tensor<1x128xf32>, tensor<f32>) -> tensor<1x128xi1>
  // CHECK: tile.select %{{.*}}, %[[cst]], %{{.*}} : (tensor<1x128xi1>, tensor<f32>, tensor<1x128xf32>) -> tensor<1x128xf32>
  // CHECK: tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}, #map{{[0-9]+}}]} : tensor<f32>, tensor<1x128xf32>, tensor<128x100xf32> -> tensor<1x100xf32>
  // CHECK: tile.add %{{.*}}, %{{.*}} : (tensor<1x100xf32>, tensor<100xf32>) -> tensor<1x100xf32>
  // CHECK: tile.contract max, none,  %[[cst_0]], %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}]} : tensor<f32>, tensor<1x100xf32> -> tensor<1x1xf32>
  // CHECK: tile.sub %{{.*}}, %{{.*}} : (tensor<1x100xf32>, tensor<1x1xf32>) -> tensor<1x100xf32>
  // CHECK: tile.exp %{{.*}} : (tensor<1x100xf32>) -> tensor<1x100xf32>
  // CHECK: tile.contract add, none, %[[cst]], %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}]} : tensor<f32>, tensor<1x100xf32> -> tensor<1x1xf32>
  // CHECK: tile.div %{{.*}}, %{{.*}} : (tensor<1x100xf32>, tensor<1x1xf32>) -> tensor<1x100xf32>
  // CHECK: return %{{.*}} : tensor<1x100xf32>
  // clang-format on
  runProgram(program);
}

Tensor Normalize(Tensor X) {
  auto XSqr = X * X;
  std::vector<TensorIndex> idxs(X.rank());
  Tensor X_MS = Contraction().sum(XSqr(idxs));
  return sqrt(X_MS);
}

std::tuple<Tensor, Tensor> LarsMomentum(  //
    Tensor X,                             //
    Tensor Grad,                          //
    Tensor Veloc,                         //
    Tensor LR,                            //
    double lars_coeff,                    //
    double lars_weight_decay,             //
    double momentum) {
  auto XNorm = Normalize(X);
  auto GradNorm = Normalize(Grad);
  auto LocLR = LR * lars_coeff * XNorm / (GradNorm + lars_weight_decay * XNorm);
  auto NewVeloc = momentum * Veloc + LocLR * (Grad + lars_weight_decay * X);
  return std::make_tuple(X - NewVeloc, NewVeloc);
}

TEST_F(CppEdsl, LarsMomentum4d) {
  auto X_shape = TensorShape(DType::FLOAT32, {4, 7, 3, 9});
  auto LR_shape = TensorShape(DType::FLOAT32, {});
  auto X = Placeholder(X_shape);
  auto Grad = Placeholder(X_shape);
  auto Veloc = Placeholder(X_shape);
  auto LR = Placeholder(LR_shape);
  auto R = LarsMomentum(X, Grad, Veloc, LR, 1. / 1024., 1. / 2048., 1. / 8.);
  auto program = makeProgram("lars_momentum4d", {X, Grad, Veloc, LR}, {std::get<0>(R), std::get<1>(R)});
  // clang-format off
  // CHECK-LABEL: CppEdsl.LarsMomentum4d
  // CHECK: module @lars_momentum4d
  // CHECK: tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK: tile.constant(1.250000e-01 : f64) : tensor<f32>
  // CHECK: tile.constant(9.765625E-4 : f64) : tensor<f32>
  // CHECK: tile.constant(4.8828125E-4 : f64) : tensor<f32>
  // CHECK: tile.mul %{{.*}}, %{{.*}} : (tensor<4x7x3x9xf32>, tensor<f32>) -> tensor<4x7x3x9xf32>
  // CHECK: tile.mul %{{.*}}, %{{.*}} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: tile.mul %{{.*}}, %{{.*}} : (tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) -> tensor<4x7x3x9xf32>
  // CHECK: tile.contract add, none, %{{.*}}, %{{.*}} {sink = #{{.*}}, srcs = [#{{.*}}]} : tensor<f32>, tensor<4x7x3x9xf32> -> tensor<f32>
  // CHECK: tile.sqrt %{{.*}} : (tensor<f32>) -> tensor<f32>
  // CHECK: tile.mul %{{.*}}, %{{.*}} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: tile.mul %{{.*}}, %{{.*}} : (tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) -> tensor<4x7x3x9xf32>
  // CHECK: tile.contract add, none, %{{.*}}, %{{.*}} {sink = #{{.*}}, srcs = [#{{.*}}]} : tensor<f32>, tensor<4x7x3x9xf32> -> tensor<f32>
  // CHECK: tile.sqrt %{{.*}} : (tensor<f32>) -> tensor<f32>
  // CHECK: tile.mul %{{.*}}, %{{.*}} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: tile.add %{{.*}}, %{{.*}} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: tile.div %{{.*}}, %{{.*}} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: tile.mul %{{.*}}, %{{.*}} : (tensor<4x7x3x9xf32>, tensor<f32>) -> tensor<4x7x3x9xf32>
  // CHECK: tile.add %{{.*}}, %{{.*}} : (tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) -> tensor<4x7x3x9xf32>
  // CHECK: tile.mul %{{.*}}, %{{.*}} : (tensor<f32>, tensor<4x7x3x9xf32>) -> tensor<4x7x3x9xf32>
  // CHECK: tile.add %{{.*}}, %{{.*}} : (tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) -> tensor<4x7x3x9xf32>
  // CHECK: tile.sub %{{.*}}, %{{.*}} : (tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) -> tensor<4x7x3x9xf32>
  // CHECK: return %{{.*}}, %{{.*}} : tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>
  // clang-format on
  runProgram(program);
}

TEST_F(CppEdsl, RepeatElements) {
  auto I = Placeholder(DType::FLOAT32, {10, 10, 10});
  TensorDim N0, N1, N2;
  TensorIndex n0, n1, n2, k;
  I.bind_dims(N0, N1, N2);
  Tensor O = Contraction()  //
                 .outShape(N0, 3 * N1, N2)
                 .outAccess(n0, 3 * n1 + k, n2)
                 .assign(I(n0, n1, n2))
                 .add_constraint(k < 3);
  auto program = makeProgram("repeat_elts", {I}, {O});
  // clang-format off
  // CHECK-LABEL: CppEdsl.RepeatElements
  // CHECK: module @repeat_elts
  // CHECK: %[[cst:.*]] = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK: tile.contract assign, none, %[[cst]], %{{.*}} {cons = #set{{[0-9]+}}, sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}]} : tensor<f32>, tensor<10x10x10xf32> -> tensor<10x30x10xf32>
  // CHECK: return %{{.*}} : tensor<10x30x10xf32>
  // clang-format on
  runProgram(program);
}

TEST_F(CppEdsl, UseDefault) {
  auto P = Placeholder(DType::FLOAT32, {1, 7, 10, 10});
  auto I = Placeholder(DType::FLOAT32, {1, 10, 10});
  TensorDim B, N1, N2;
  TensorIndex b, i1, i2;
  I.bind_dims(B, N1, N2);
  Tensor O = Contraction().outShape(B, 7, N1, N2).outAccess(b, 3, i1, i2).assign(I(b, i1, i2)).init(P);
  auto program = makeProgram("use_default", {I, P}, {O});
  // clang-format off
  // CHECK-LABEL: CppEdsl.UseDefault
  // CHECK: module @use_default
  // CHECK: tile.contract assign, none, %{{.*}}, %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}]} : tensor<1x7x10x10xf32>, tensor<1x10x10xf32> -> tensor<1x7x10x10xf32>
  // CHECK: return %{{.*}} : tensor<1x7x10x10xf32>
  // clang-format on
  runProgram(program);
}

TEST_F(CppEdsl, UniqueNames) {
  TensorShape shape(DType::FLOAT32, {1});
  auto A = Placeholder(shape, "A");
  auto B = Placeholder(shape, "B");
  auto C0 = Placeholder(shape, "C");
  auto C1 = Placeholder(shape, "C");
  auto program = makeProgram("unique_names", {A, B, C0, C1}, {A + B + C0 + C1});
  // clang-format off
  // CHECK-LABEL: CppEdsl.UniqueNames
  // CHECK: module @unique_names
  // CHECK: tile.add %{{.*}}, %{{.*}} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  // CHECK: tile.add %{{.*}}, %{{.*}} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  // CHECK: tile.add %{{.*}}, %{{.*}} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  // CHECK: return %{{.*}} : tensor<1xf32>
  // clang-format on
  runProgram(program);
}

TEST_F(CppEdsl, GlobalMin) {
  auto I = Placeholder(DType::FLOAT32, {10, 10, 10}, "I");
  TensorIndex i, j, k;
  Tensor I_Neg = -I;
  Tensor O = -Contraction().max(I_Neg(i, j, k)).build();
  auto program = makeProgram("global_min", {I}, {O});
  // clang-format off
  // CHECK-LABEL: CppEdsl.GlobalMin
  // CHECK: module @global_min
  // CHECK: %[[cst:.*]] = tile.constant(0xFFF0000000000000 : f64) : tensor<f32>
  // CHECK: tile.neg %{{.*}} : (tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  // CHECK: tile.contract max, none, %[[cst]], %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}]} : tensor<f32>, tensor<10x10x10xf32> -> tensor<f32>
  // CHECK: tile.neg %{{.*}} : (tensor<f32>) -> tensor<f32>
  // CHECK: return %{{.*}} : tensor<f32>
  // clang-format on
  runProgram(program);
}

TEST_F(CppEdsl, CumSum) {
  auto I = Placeholder(DType::FLOAT32, {10}, "I");
  TensorDim N;
  TensorIndex i, k;
  I.bind_dims(N);
  Tensor O = Contraction().outShape(N).outAccess(i).sum(I(k)).add_constraint(i - k < N);
  auto program = makeProgram("cumsum", {I}, {O});
  // clang-format off
  // CHECK-LABEL: CppEdsl.CumSum
  // CHECK: module @cumsum
  // CHECK: %[[cst:.*]] = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK: tile.contract add, none, %[[cst]], %{{.*}} {cons = #set{{[0-9]+}}, sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}]} : tensor<f32>, tensor<10xf32> -> tensor<10xf32>
  // CHECK: return %{{.*}} : tensor<10xf32>
  // clang-format on
  runProgram(program);
}

Tensor ComplexConv2d(Tensor I, Tensor K,
                     const std::vector<size_t>& s,  // stride coeffs
                     const std::vector<size_t>& d   // dilation coeffs
) {
  // "same-lower" autopadding will be applied
  TensorDim N, G, GCI, GCO;
  std::vector<TensorDim> X(2);
  std::vector<TensorDim> KX(2);
  TensorIndex n, g, gci, gco;
  std::vector<TensorIndex> x(2);
  std::vector<TensorIndex> k(2);
  I.bind_dims(N, X[0], X[1], G, GCI);
  K.bind_dims(KX[0], KX[1], G, GCI, GCO);
  // Compute output spatial dimensions
  std::vector<TensorDim> Y(2);
  for (size_t i = 0; i < Y.size(); ++i) {
    Y[i] = (X[i] + s[i] - 1) / s[i];
  }
  // Compute the effective kernel size after dilation
  std::vector<TensorDim> EK(2);
  for (size_t i = 0; i < EK.size(); ++i) {
    EK[i] = d[i] * (KX[i] - 1) + 1;
  }
  // Compute the padding offset
  std::vector<TensorDim> P(2);
  for (size_t i = 0; i < P.size(); ++i) {
    P[i] = ((Y[i] - 1) * s[i] + EK[i] - X[i]) / 2;
  }
  // Compute the convolution
  return Contraction()
      .outShape(N, Y[0], Y[1], G, GCO)
      .outAccess(n, x[0], x[1], g, gco)
      .sum(I(n, s[0] * x[0] + d[0] * k[0] - P[0], s[1] * x[1] + d[1] * k[1] - P[1], g, gci) *
           K(k[0], k[1], g, gci, gco));
}

TEST_F(CppEdsl, ComplexConv2d) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3, 3});
  auto K = Placeholder(DType::FLOAT32, {3, 3, 3, 3, 32});
  auto O = ComplexConv2d(I, K, {2, 2}, {3, 3});
  auto program = makeProgram("complex_conv_2d", {I, K}, {O});
  // clang-format off
  // CHECK-LABEL: CppEdsl.ComplexConv2d
  // CHECK: module @complex_conv_2d
  // CHECK: %[[cst:.*]] = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK: tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}, #map{{[0-9]+}}]} : tensor<f32>, tensor<1x224x224x3x3xf32>, tensor<3x3x3x3x32xf32> -> tensor<1x112x112x3x32xf32>
  // CHECK: return %{{.*}} : tensor<1x112x112x3x32xf32>
  // clang-format on
  runProgram(program);
}

TEST_F(CppEdsl, Reciprocal) {
  auto A = Placeholder(DType::FLOAT32, {6}, "A");
  auto R = 1.0 / A;
  auto program = makeProgram("reciprocal", {A}, {R});
  // clang-format off
  // CHECK-LABEL: CppEdsl.Reciprocal
  // CHECK: module @reciprocal
  // CHECK: %[[cst:.*]] = tile.constant(1.000000e+00 : f64) : tensor<f32>
  // CHECK: tile.div %[[cst]], %{{.*}} : (tensor<f32>, tensor<6xf32>) -> tensor<6xf32>
  // CHECK: return %{{.*}} : tensor<6xf32>
  // clang-format on
  std::vector<float> input = {1, 2, 4, 5, 8, 10};
  std::vector<float> expected = {1.0, 0.5, 0.25, 0.2, 0.125, 0.1};
  checkExact(program, {input}, {expected});
}

TEST_F(CppEdsl, ReshapeFold) {
  auto A = Placeholder(DType::INT32, {3, 3}, "A");
  auto R = reshape(A, {3, 3});
  auto program = makeProgram("reshape_fold", {A}, {R});
  // clang-format off
  // CHECK-LABEL: CppEdsl.ReshapeFold
  // CHECK: module @reshape_fold
  // CHECK: %[[X0:.*]] = tile.ident %{{.*}} : (tensor<3x3xsi32>) -> tensor<3x3xsi32>
  // CHECK-NEXT: return %[[X0]]
  // clang-format on
  std::vector<int32_t> input = {
      1, 2, 3,  //
      4, 5, 6,  //
      7, 8, 9,  //
  };
  checkExact(program, {input}, {input});
}

TEST_F(CppEdsl, ReshapeScalar) {
  auto A = Placeholder(DType::INT32, {}, "A");
  std::vector<int64_t> shape = {};
  auto R = reshape(A, shape);
  auto program = makeProgram("reshape_scalar", {A}, {R});
  // clang-format off
  // CHECK-LABEL: CppEdsl.ReshapeScalar
  // CHECK: module @reshape_scalar
  // CHECK: %[[X0:.*]] = tile.ident %{{.*}} : (tensor<si32>) -> tensor<si32>
  // CHECK-NEXT: return %[[X0]] : tensor<si32>
  // clang-format on
  std::vector<int32_t> data = {2};
  checkExact(program, {data}, {data});
}

TEST_F(CppEdsl, ReshapeIntoScalar) {
  auto A = Placeholder(DType::INT32, {1, 1, 1}, "A");
  std::vector<int64_t> shape = {};
  auto R = reshape(A, shape);
  auto program = makeProgram("reshape_into_scalar", {A}, {R});
  // clang-format off
  // CHECK-LABEL: CppEdsl.ReshapeIntoScalar
  // CHECK: module @reshape_into_scalar
  // CHECK:      %[[X0:.*]] = tile.reshape %{{.*}} : (tensor<1x1x1xsi32>) -> tensor<si32>
  // CHECK-NEXT: %[[X1:.*]] = tile.ident %[[X0]] : (tensor<si32>) -> tensor<si32>
  // CHECK-NEXT: return %[[X1]] : tensor<si32>
  // clang-format on

  std::vector<int32_t> data = {2};
  checkExact(program, {data}, {data});
}

TEST_F(CppEdsl, ReshapeFromScalar) {
  auto A = Placeholder(DType::INT32, {}, "A");
  std::vector<int64_t> shape = {1, 1, 1};
  auto R = reshape(A, shape);
  auto program = makeProgram("reshape_from_scalar", {A}, {R});
  // clang-format off
  // CHECK-LABEL: CppEdsl.ReshapeFromScalar
  // CHECK: module @reshape_from_scalar
  // CHECK:      %[[X0:.*]] = tile.reshape %{{.*}} : (tensor<si32>) -> tensor<1x1x1xsi32>
  // CHECK-NEXT: %[[X1:.*]] = tile.ident %[[X0]] : (tensor<1x1x1xsi32>) -> tensor<1x1x1xsi32>
  // CHECK-NEXT: return %[[X1]] : tensor<1x1x1xsi32>
  // clang-format on
  std::vector<int32_t> data = {2};
  checkExact(program, {data}, {data});
}

TEST_F(CppEdsl, DefractLong) {
  std::vector<int64_t> input_shape{1, 3, 3, 1};
  std::vector<int64_t> output_shape{1, 5, 5, 1};
  auto I = Placeholder(DType::FLOAT32, input_shape, "I");
  auto K = Placeholder(DType::FLOAT32, input_shape, "K");
  TensorIndex n, x0, x1, k0, k1, co, ci;
  Tensor O = Contraction()
                 .outShape(output_shape)
                 .outAccess(n, x0, x1, co)
                 .sum(I(n, (x0 + k0 - 1) / 2, (x1 + k1 - 1) / 2, ci) * K(2 - k0, 2 - k1, co, ci));
  auto program = makeProgram("defract_long", {I, K}, {O});
  // clang-format off
  // CHECK-LABEL: CppEdsl.DefractLong
  // CHECK: module @defract_long
  // CHECK: %[[cst:.*]] = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK: tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}, #map{{[0-9]+}}]} : tensor<f32>, tensor<1x3x3x1xf32>, tensor<1x3x3x1xf32> -> tensor<1x5x5x1xf32>
  // CHECK: return %{{.*}} : tensor<1x5x5x1xf32>
  // clang-format on
  runProgram(program);
}

TEST_F(CppEdsl, DupOut) {
  auto A = Placeholder(DType::FLOAT32, {10, 20});
  auto B = Placeholder(DType::FLOAT32, {20, 30});
  auto C = Placeholder(DType::FLOAT32, {30, 40});
  auto R = Dot(Dot(A, B), C);
  auto program = makeProgram("dup_out", {A, B, C}, {R, R, R});
  // clang-format off
  // CHECK: module @dup_out
  // CHECK: %[[cst:.*]] = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK: tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}, #map{{[0-9]+}}]} : tensor<f32>, tensor<10x20xf32>, tensor<20x30xf32> -> tensor<10x30xf32>
  // CHECK: %[[out:.*]] = tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}, #map{{[0-9]+}}]} : tensor<f32>, tensor<10x30xf32>, tensor<30x40xf32> -> tensor<10x40xf32>
  // CHECK: %[[i2:.*]] = tile.ident %[[out]] : (tensor<10x40xf32>) -> tensor<10x40xf32>
  // CHECK: %[[i3:.*]] = tile.ident %[[out]] : (tensor<10x40xf32>) -> tensor<10x40xf32>
  // CHECK: return %[[out]], %[[i2]], %[[i3]] : tensor<10x40xf32>, tensor<10x40xf32>, tensor<10x40xf32>
  // clang-format on
  runProgram(program);
}

TEST_F(CppEdsl, Select) {
  auto I = Placeholder(DType::FLOAT32, {10, 20});
  auto one = cast(Tensor{1.0}, DType::FLOAT32);
  auto zero = cast(Tensor{0.0}, DType::FLOAT32);
  auto O = select(I == 0, zero, one);
  auto program = makeProgram("select", {I}, {O});
  // clang-format off
  // CHECK-LABEL: CppEdsl.Select
  // CHECK: module @select
  // CHECK-DAG: %[[fp1:.*]] = tile.constant(1.000000e+00 : f64) : tensor<f32>
  // CHECK-DAG: %[[fp0:.*]] = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK: tile.cmp_eq %{{.*}}, %[[fp0]] : (tensor<10x20xf32>, tensor<f32>) -> tensor<10x20xi1>
  // CHECK: tile.select %{{.*}}, %[[fp0]], %[[fp1]] : (tensor<10x20xi1>, tensor<f32>, tensor<f32>) -> tensor<10x20xf32>
  // CHECK: return %{{.*}} : tensor<10x20xf32>
  // clang-format on
  runProgram(program);
}

TEST_F(CppEdsl, Shape) {
  auto I = Placeholder(DType::FLOAT32, {2, 3});
  auto O = shape(I);
  auto program = makeProgram("shape", {I}, {O});
  // clang-format off
  // CHECK-LABEL: CppEdsl.Shape
  // CHECK: module @shape
  // CHECK: tile.shape %{{.*}} : (tensor<2x3xf32>) -> tensor<2xsi32>
  // CHECK: return %{{.*}} : tensor<2xsi32>
  // clang-format on
  std::vector<float> input = {
      1, 2, 3,  //
      1, 2, 3,  //
  };
  std::vector<int32_t> expected = {2, 3};
  checkExact(program, {input}, {expected});
}

TEST_F(CppEdsl, Prng) {
  auto S = Placeholder(DType::UINT32, {1, 3});
  auto [O, NS] = prng(S, {2, 3});  // NOLINT
  auto program = makeProgram("prng", {S}, {O, NS});
  // clang-format off
  // CHECK-LABEL: CppEdsl.Prng
  // CHECK: module @prng
  // CHECK: %result, %new_state = tile.prng %{{.*}} : (tensor<1x3xui32>) -> (tensor<2x3xf32>, tensor<1x3xui32>)
  // CHECK: return %result, %new_state : tensor<2x3xf32>, tensor<1x3xui32>
  // clang-format on

  std::vector<uint32_t> state = {
      5, 6, 7,  //
  };

  std::vector<float> result = {
      9.31323e-10, 3.8147e-06,  0.0156251,  //
      0.000244171, 0.000125885, 0.515625,   //
  };

  std::vector<uint32_t> new_state = {
      1052804, 0, 0  //
  };

  checkClose(program, {state}, {result, new_state});
}

TEST_F(CppEdsl, Cos) {
  auto S = Placeholder(DType::FLOAT32, {3, 3});
  auto O = cos(S);
  auto program = makeProgram("cos", {S}, {O});
  std::vector<float> A_input = {
      5.0, 6.0, 7.0,  //
      4.0, 5.0, 6.0,  //
      7.0, 8.0, 9.0,  //
  };

  std::vector<float> C_output = {0.283662, 0.96017,  0.753902, -0.653644, 0.283662,
                                 0.96017,  0.753902, -0.1455,  -0.91113};
  checkClose(program, {A_input}, {C_output}, /*tolerance=*/1e-4);
}

TEST_F(CppEdsl, Sin) {
  auto S = Placeholder(DType::FLOAT32, {3, 3});
  auto O = sin(S);
  auto program = makeProgram("sin", {S}, {O});
  std::vector<float> A_input = {
      5.0, 6.0,  7.0,  //
      4.0, -5.0, 1.1,  //
      7.0, 8.0,  9.0,  //
  };

  std::vector<float> C_output = {
      -0.958924, -0.279415, 0.656987,  //
      -0.756802, 0.958924,  0.891207,  //
      0.656987,  0.989358,  0.412118   //
  };
  checkClose(program, {A_input}, {C_output}, /*tolerance=*/1e-4);
}

TEST_F(CppEdsl, ConvI8) {
  auto I = Placeholder(DType::INT8, {1, 224, 224, 3});
  auto K = Placeholder(DType::INT8, {3, 3, 3, 32});
  auto O = Convolution2(I, K);
  auto program = makeProgram("convolution", {I, K}, {O});
  // clang-format off
  // CHECK-LABEL: CppEdsl.ConvI8
  // CHECK: module @convolution
  // CHECK: %[[cst:.*]] = tile.constant(0 : i64) : tensor<si8>
  // CHECK: tile.contract add, mul, %[[cst]], %{{.*}}, %{{.*}} {sink = #map{{[0-9]+}}, srcs = [#map{{[0-9]+}}, #map{{[0-9]+}}]} : tensor<si8>, tensor<1x224x224x3xsi8>, tensor<3x3x3x32xsi8> -> tensor<1x224x224x32xsi8>
  // CHECK: return %{{.*}} : tensor<1x224x224x32xsi8>
  // clang-format on
  runProgram(program);
}

TEST_F(CppEdsl, LogicalAnd_uint64) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = Placeholder(DType::UINT64, {3, 3});
  auto C = A && B;
  auto program = makeProgram("logical_and", {A, B}, {C});

  std::vector<uint64_t> A_input{1, 2, 3,  //
                                4, 0, 6,  //
                                7, 0, 9};
  std::vector<uint64_t> B_input{10, 11, 12,  //
                                13, 14, 15,  //
                                16, 17, 18};
  std::vector<int8_t> C_output{1, 1, 1,  //
                               1, 0, 1,  //
                               1, 0, 1};
  checkExact(program, {A_input, B_input}, {C_output});
}

TEST_F(CppEdsl, LogicalAnd_mixed) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = Placeholder(DType::FLOAT32, {3, 3});
  auto C = A && B;
  auto program = makeProgram("logical_and", {A, B}, {C});

  std::vector<uint64_t> A_input{1, 2, 3,  //
                                4, 0, 6,  //
                                7, 0, 9};
  std::vector<float> B_input{10.0, 11.0, 12.0,  //
                             13.0, 14.0, 15.0,  //
                             16.0, 17.0, 18.0};
  std::vector<int8_t> C_output{1, 1, 1,  //
                               1, 0, 1,  //
                               1, 0, 1};
  checkExact(program, {A_input, B_input}, {C_output});
}

TEST_F(CppEdsl, LogicalOr_uint64) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = Placeholder(DType::UINT64, {3, 3});
  auto C = A || B;
  auto program = makeProgram("logical_or", {A, B}, {C});

  std::vector<uint64_t> A_input{1, 2, 3,  //
                                4, 0, 6,  //
                                7, 0, 9};
  std::vector<uint64_t> B_input{10, 11, 12,  //
                                0,  0,  0,   //
                                16, 17, 18};
  std::vector<int8_t> C_output{1, 1, 1,  //
                               1, 0, 1,  //
                               1, 1, 1};
  checkExact(program, {A_input, B_input}, {C_output});
}

TEST_F(CppEdsl, LogicalOr_float) {
  auto A = Placeholder(DType::FLOAT32, {3, 3});
  auto B = Placeholder(DType::FLOAT32, {3, 3});
  auto C = A || B;
  auto program = makeProgram("logical_or", {A, B}, {C});

  std::vector<float> A_input{1.0, 2.0, 3.0,  //
                             4.0, 0.0, 6.0,  //
                             7.0, 0.0, 9.0};
  std::vector<float> B_input{10.0, 11.0, 12.0,  //
                             0.0,  0.0,  0.0,   //
                             16.0, 17.0, 18.0};
  std::vector<int8_t> C_output{1, 1, 1,  //
                               1, 0, 1,  //
                               1, 1, 1};
  checkExact(program, {A_input, B_input}, {C_output});
}

TEST_F(CppEdsl, LogicalOr_int32) {
  auto A = Placeholder(DType::INT32, {3, 3});
  auto B = Placeholder(DType::INT32, {3, 3});
  auto C = A || B;
  auto program = makeProgram("logical_or", {A, B}, {C});

  std::vector<int32_t> A_input{1, 2, 3,  //
                               4, 0, 6,  //
                               7, 0, 9};
  std::vector<int32_t> B_input{10, 11, 12,  //
                               0,  0,  0,   //
                               16, 17, 18};
  std::vector<int8_t> C_output{1, 1, 1,  //
                               1, 0, 1,  //
                               1, 1, 1};
  checkExact(program, {A_input, B_input}, {C_output});
}

TEST_F(CppEdsl, LogicalNot_int32) {
  auto A = Placeholder(DType::INT32, {3, 3});
  auto R = !A;
  auto program = makeProgram("logical_not", {A}, {R});

  std::vector<int32_t> input{1, 2, 3,  //
                             4, 0, 6,  //
                             7, 0, 9};
  std::vector<int8_t> expected{0, 0, 0,  //
                               0, 1, 0,  //
                               0, 1, 0};
  checkExact(program, {input}, {expected});
}

TEST_F(CppEdsl, LogicalNot_float) {
  auto A = Placeholder(DType::FLOAT32, {3, 3});
  auto R = !A;
  auto program = makeProgram("logical_not", {A}, {R});

  std::vector<float> input{1.0, 2.0, 3.0,  //
                           4.0, 0,   6.5,  //
                           7.7, 0,   0.9};
  std::vector<int8_t> expected{0, 0, 0,  //
                               0, 1, 0,  //
                               0, 1, 0};
  checkExact(program, {input}, {expected});
}

TEST_F(CppEdsl, Asin) {
  auto S = Placeholder(DType::FLOAT32, {3, 3});
  auto O = asin(S);
  auto program = makeProgram("asin", {S}, {O});

  std::vector<float> input = {
      0.1, 0.2, 0.3,   //
      0.4, 0.5, 0.6,   //
      1.0, 0.0, -0.6,  //
  };
  std::vector<float> expected = {
      0.100167, 0.201358, 0.304693,  //
      0.411517, 0.523599, 0.643501,  //
      1.5708,   0.0,      -0.643501  //
  };
  checkClose(program, {input}, {expected});
}

TEST_F(CppEdsl, Acos) {
  auto S = Placeholder(DType::FLOAT32, {3, 3});
  auto O = acos(S);
  auto program = makeProgram("acos", {S}, {O});

  std::vector<float> input = {
      0.1, 0.2, 0.3,   //
      0.4, 0.5, 0.6,   //
      1.0, 0.0, -0.6,  //
  };
  std::vector<float> expected = {
      1.47063, 1.36944, 1.2661,    //
      1.15928, 1.0472,  0.927295,  //
      0.0,     1.5708,  2.2143     //
  };
  checkClose(program, {input}, {expected}, /*tolerance=*/1e-4);
}

TEST_F(CppEdsl, Atan) {
  auto S = Placeholder(DType::FLOAT32, {3, 3});
  auto O = atan(S);
  auto program = makeProgram("atan", {S}, {O});

  std::vector<float> input = {
      0.1, 0.2, 0.3,   //
      0.4, 0.5, 0.6,   //
      1.0, 0.0, -0.6,  //
  };
  std::vector<float> expected = {
      0.0996687, 0.197396, 0.291457,  //
      0.380506,  0.463648, 0.54042,   //
      0.785398,  0,        -0.54042   //
  };
  checkClose(program, {input}, {expected});
}

TEST_F(CppEdsl, CosH) {
  auto S = Placeholder(DType::FLOAT32, {3, 3});
  auto O = cosh(S);
  auto program = makeProgram("cosh", {S}, {O});

  std::vector<float> input = {
      0.1, 0.2, 0.3,   //
      0.4, 0.5, 0.6,   //
      1.0, 0.0, -0.6,  //
  };
  std::vector<float> expected = {
      1.005,   1.02007, 1.04534,  //
      1.08107, 1.12763, 1.18547,  //
      1.54308, 1,       1.18547   //
  };
  checkClose(program, {input}, {expected});
}

TEST_F(CppEdsl, Erf) {
  auto S = Placeholder(DType::FLOAT32, {3, 3});
  auto O = erf(S);
  auto program = makeProgram("erf", {S}, {O});

  std::vector<float> input = {
      0.1, 0.2, 0.3,   //
      0.4, 0.5, 0.6,   //
      1.0, 0.0, -0.6,  //
  };
  std::vector<float> expected = {
      0.112463, 0.222703, 0.328627,  //
      0.428392, 0.5205,   0.603856,  //
      0.842701, 0,        -0.603856  //
  };
  checkClose(program, {input}, {expected});
}

TEST_F(CppEdsl, Floor) {
  auto S = Placeholder(DType::FLOAT32, {3, 3});
  auto O = floor(S);
  auto program = makeProgram("floor", {S}, {O});

  std::vector<float> input = {
      1.1,  9.21, 3.0,   //
      -0.4, -7.0, 0.6,   //
      1.0,  0.0,  -6.6,  //
  };
  std::vector<float> expected = {
      1,  9,  3,  //
      -1, -7, 0,  //
      1,  0,  -7  //
  };
  checkExact(program, {input}, {expected});
}

TEST_F(CppEdsl, Pow) {
  auto A = Placeholder(DType::FLOAT32, {3, 3});
  auto B = Placeholder(DType::FLOAT32, {3, 3});
  auto O = pow(A, B);
  auto program = makeProgram("pow", {A, B}, {O});

  std::vector<float> A_input = {
      0.5, 1.5, 2.5,  //
      0.5, 1.5, 2.5,  //
      0.5, 1.5, 2.5,  //
  };
  std::vector<float> B_input = {
      0.5, 0.5, 0.5,  //
      1.5, 1.5, 1.5,  //
      2.5, 2.5, 2.5,  //
  };
  std::vector<float> expected = {
      0.707107, 1.22474, 1.58114,  //
      0.353553, 1.83712, 3.95285,  //
      0.176777, 2.75568, 9.88212   //
  };
  checkClose(program, {A_input, B_input}, {expected});
}

TEST_F(CppEdsl, Round) {
  auto S = Placeholder(DType::FLOAT32, {3, 3});
  auto O = round(S);
  auto program = makeProgram("round", {S}, {O});

  std::vector<float> input = {
      1.1,  9.21, 3.0,   //
      -0.4, -7.0, 0.6,   //
      1.0,  0.0,  -6.6,  //
  };
  std::vector<float> expected = {
      1,  9,  3,  //
      -0, -7, 1,  //
      1,  0,  -7  //
  };
  checkExact(program, {input}, {expected});
}

TEST_F(CppEdsl, SinH) {
  auto S = Placeholder(DType::FLOAT32, {3, 3});
  auto O = sinh(S);
  auto program = makeProgram("sinh", {S}, {O});

  std::vector<float> input = {
      -2.0, -1.5, 0.0,  //
      -1.0, 0.1,  0.2,  //
      1.0,  1.5,  5.0,  //
  };
  std::vector<float> expected = {
      -3.62686, -2.12928, 0,         //
      -1.1752,  0.100167, 0.201336,  //
      1.1752,   2.12928,  74.2032    //
  };
  checkClose(program, {input}, {expected});
}

TEST_F(CppEdsl, Tan) {
  auto S = Placeholder(DType::FLOAT32, {3, 3});
  auto O = tan(S);
  auto program = makeProgram("tan", {S}, {O});

  std::vector<float> input = {
      0.1, 0.2, 0.3,   //
      0.4, 0.5, 0.6,   //
      1.0, 0.0, -0.6,  //
  };
  std::vector<float> expected = {
      0.100335, 0.20271,  0.309336,  //
      0.422793, 0.546302, 0.684137,  //
      1.55741,  0,        -0.684137  //
  };
  checkClose(program, {input}, {expected}, /*tolerance=*/1e-4);
}

TEST_F(CppEdsl, Scatter1D) {
  auto I = Placeholder(DType::INT32, {4});
  auto U = Placeholder(DType::FLOAT32, {4});
  auto S = Placeholder(DType::INT32, {8});
  auto O = scatter(U, I, S);
  auto program = makeProgram("scatter", {I, U, S}, {O});

  std::vector<int32_t> indices = {4, 3, 1, 7};
  std::vector<float> updates = {9, 10, 11, 12};
  std::vector<int32_t> shape = {8};
  std::vector<float> expected = {0, 11, 0, 10, 9, 0, 0, 12};
  checkExact(program, {indices, updates, shape}, {expected});
}

TEST_F(CppEdsl, Scatter3D) {
  auto I = Placeholder(DType::INT32, {2});
  auto U = Placeholder(DType::FLOAT32, {2, 4, 4});
  auto S = Placeholder(DType::INT32, {4, 4, 4});
  auto O = scatter(U, I, S);
  auto program = makeProgram("scatter", {I, U, S}, {O});

  std::vector<int32_t> indices = {0, 2};
  std::vector<float> updates = {
      5, 5, 5, 5, 6, 6, 6, 6,  //
      7, 7, 7, 7, 8, 8, 8, 8,  //
      5, 5, 5, 5, 6, 6, 6, 6,  //
      7, 7, 7, 7, 8, 8, 8, 8   //
  };
  std::vector<int32_t> shape = {4, 4, 4};
  std::vector<float> expected = {
      5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,  //
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //
      5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,  //
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   //
  };
  checkExact(program, {indices, updates, shape}, {expected});
}

TEST_F(CppEdsl, Trace) {
  auto I = Placeholder(DType::FLOAT32, {3, 3});
  auto O = trace(I, "msg");
  auto program = makeProgram("trace", {I}, {O});
  // clang-format off
  // CHECK-LABEL: CppEdsl.Trace
  // CHECK: module @trace
  // CHECK: tile.pragma %{{.*}} "trace" {msg = "msg"} : tensor<3x3xf32>
  // CHECK: return %{{.*}} : tensor<3x3xf32>
  // clang-format on
}

Tensor Transpose(Tensor I, const std::string& layout) {
  TensorLens lens(layout, "MN");
  I = I.use(lens);
  TensorDim M, N;
  TensorIndex i, j;
  I.bind_dims(M, N);
  return Contraction(lens).outShape(N, M).outAccess(j, i).assign(I(i, j));
}

TEST_F(CppEdsl, Lens) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3});
  auto K = Placeholder(DType::FLOAT32, {3, 3, 3, 32});
  auto O = Convolution2(I, K, /*I_layout=*/"NHWC", /*K_layout=*/"HWCK");
  makeProgram("conv2d_nhwc", {I, K}, {O});

  I = Placeholder(DType::FLOAT32, {1, 3, 224, 224});
  K = Placeholder(DType::FLOAT32, {3, 32, 7, 7});
  O = Convolution2(I, K, /*I_layout=*/"NCHW", /*K_layout=*/"CKHW");
  makeProgram("conv2d_nchw", {I, K}, {O});

  std::vector<float> input = {
      1, 2, 3,  //
      4, 5, 6,  //
  };

  std::vector<float> expected = {
      1, 4,  //
      2, 5,  //
      3, 6,  //
  };

  I = Placeholder(DType::FLOAT32, {2, 3});
  O = Transpose(I, "MN");
  auto program = makeProgram("transpose_mn", {I}, {O});
  checkExact(program, {input}, {expected});

  I = Placeholder(DType::FLOAT32, {2, 3});
  O = Transpose(I, "NM");
  program = makeProgram("transpose_nm", {I}, {O});
  checkExact(program, {input}, {expected});
}

TEST_F(CppEdsl, Layer) {
  auto A = Placeholder(DType::FLOAT32, {10, 20});
  Tensor O = layer("relu", [&]() {  //
    return Relu(A);
  });
  auto program = makeProgram("relu", {A}, {O});
  // clang-format off
  // CHECK-LABEL: CppEdsl.Layer
  // CHECK: module @relu
  // CHECK: %[[X0:.*]] = layer.box "relu" (%[[arg1:.*]]) = (%{{.*}}) : (tensor<10x20xf32>) -> tensor<10x20xf32>
  // CHECK:   %[[cst:.*]] = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK:   %[[X1:.*]] = tile.cmp_lt %[[arg1]], %[[cst]] : (tensor<10x20xf32>, tensor<f32>) -> tensor<10x20xi1>
  // CHECK:   %[[X2:.*]] = tile.select %[[X1]], %[[cst]], %[[arg1]] : (tensor<10x20xi1>, tensor<f32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  // CHECK:   layer.return %[[X2]] : tensor<10x20xf32>
  // CHECK: return %[[X0]] : tensor<10x20xf32>
  // clang-format on
  runProgram(program);
}

}  // namespace
}  // namespace plaidml::edsl
