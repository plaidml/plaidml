
// Copyright 2020 Intel Corporation.
// DO NOT TOUCH THIS FILE
// Note: This file is being used by sphinx docs to pull in code blocks.
//       Any changes made here may upset the docs.
//       Code blocks are pulled into docs/usage/writing_edsl.rst if line numbers change here
//       please update docs/usage/edsl.rst
#include <float.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"
#include "llvm/ADT/StringRef.h"

using ::testing::ContainerEq;
using ::testing::Eq;

namespace plaidml::edsl {

bool operator==(const Program& lhs, const std::string& rhs) {  //
  return llvm::StringRef(lhs.str()).trim() == llvm::StringRef(rhs).trim();
}

// sum_over_axis_start
Tensor SumOveAxis(const Tensor& I) {
  TensorDim M, N;
  TensorIndex m, n;
  I.bind_dims(M, N);
  auto O = TensorOutput(N);
  O(n) += I(m, n);  // contraction
  return O;
}
// sum_over_axis_end

// max_over_axis_start
Tensor MaxOverAxis(const Tensor& I) {
  TensorDim M, N;
  TensorIndex m, n;
  I.bind_dims(M, N);
  auto O = TensorOutput(N);
  O(n) >= I(m, n);
  return O;
}
// max_over_axis_end

// matmul_start
Tensor MatMul(const Tensor& A, const Tensor& B) {
  TensorDim I, J, K;
  TensorIndex i, j, k;
  A.bind_dims(I, K);
  B.bind_dims(K, J);
  auto C = TensorOutput(I, J);
  C(i, j) += A(i, k) * B(k, j);
  return C;
}
// matmul_end

// global_min_start
Tensor GlobalMin(const Tensor& I) {
  TensorIndex i, j, k;
  auto Neg = -I;
  auto O_Neg = TensorOutput();
  O_Neg() >= Neg(i, j, k);
  auto O = -O_Neg;
  return O;
}
// global_min_end

// avg_start
Tensor Avg(const Tensor& I) {
  TensorDim X, Y;
  TensorIndex x, y;
  I.bind_dims(X, Y);
  auto Sum = TensorOutput();
  Sum(y) += I(x, y);
  return Sum / X;
}
// avg_end

// avg_stages_start
Tensor AvgStages(const Tensor& I) {
  TensorDim X, Y;
  TensorIndex x, y;
  I.bind_dims(X, Y);
  auto Sum = TensorOutput();
  Sum() += I(x, y);
  auto PartialMean = Sum / X;
  return PartialMean / Y;
}
// avg_stages_end

// avg_merge_start
Tensor AvgMerge(const Tensor& I) {
  TensorDim X, Y;
  TensorIndex x, y;
  I.bind_dims(X, Y);
  auto Sum = TensorOutput();
  Sum() += I(x, y);
  return Sum / (X * Y);
}
// avg_merge_end

void ForLoopMaxPool1D() {
  int N = 10;
  // for_loop_max_pool_start
  float I[N], O[N / 2];
  for (int i = 0; i < N / 2; ++i) {
    float curr_max = FLT_MIN;
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
  auto O = TensorOutput(N / 2);
  O(i) >= I(2 * i + j);
  return O;
}
// wrong_max_pool_end

Tensor ValidIndices(const Tensor& I) {
  TensorDim N;
  TensorIndex i, j;
  // valid_indices_start
  I.bind_dims(N);
  auto O = TensorOutput(N / 2);
  O(i) >= I(2 * i + j);
  O.add_constraint(j < 2);
  // valid_indices_end
  return O;
}

// max_pool_1d_start
Tensor MaxPool1D(const Tensor& I) {
  TensorDim N;
  TensorIndex i, j;
  I.bind_dims(N);
  auto O = TensorOutput(N / 2);
  O(i) >= I(2 * i + j);
  O.add_constraint(j < 2);
  return O;
}
// max_pool_1d_end

// max_pool_1d_odd_start
Tensor MaxPool1DOdd(const Tensor& I) {
  TensorDim N;
  TensorIndex i, j;
  I.bind_dims(N);
  auto O = TensorOutput((N + 1) / 2);
  O(i) >= I(2 * i + j);
  O.add_constraint(j < 2);
  return O;
}
// max_pool_1d_odd_end

// skip_start
Tensor Skip(const Tensor& I) {
  TensorDim M, N;
  TensorIndex i, j;
  I.bind_dims(M, N);
  auto O = TensorOutput(N);
  O(2 * i) += I(2 * i, j);
  return O;
}
// skip_end

// cumsum_start
Tensor CumSum(const Tensor& I) {
  TensorDim N;
  TensorIndex i, k;
  I.bind_dims(N);
  auto O = TensorOutput(N);
  O(i) += I(k);
  O.add_constraint(i - k < N);
  return O;
}
// cumsum_end

// conv_1d_start
Tensor Conv1D(const Tensor& I, const Tensor& K) {
  TensorDim N, X, KX, CI, CO;
  TensorIndex n, x, k, ci, co;
  I.bind_dims(N, X, CI);
  K.bind_dims(KX, CI, CO);
  auto O = TensorOutput(N, X - KX + 1, CO);
  O(n, x, co) += I(n, x + k, ci) * K(k, ci, co);
  return O;
}
// conv_1d_end

// conv_2d_dilated_start
Tensor Conv2DDilated(const Tensor& I, const Tensor& K) {
  TensorDim N, X, Y, KX, KY, CI, CO;
  TensorIndex n, x, y, kx, ky, ci, co;
  I.bind_dims(N, X, Y, CI);
  K.bind_dims(KX, KY, CI, CO);
  auto O = TensorOutput(N, X - 2 * (KX - 1), Y - 3 * (KY - 1), CO);
  O(n, x, y, co) += I(n, x + 2 * kx, y + 3 * ky, ci) * K(kx, ky, ci, co);
  return O;
}
// conv_2d_dilated_end

// complex_conv_start
Tensor ComplexConv2D(const Tensor& I, const Tensor& K,
                     const std::vector<size_t>& s,  // stride coeffs
                     const std::vector<size_t>& d   // dilation coeffs
) {
  // "same-lower" autopadding will be applied
  TensorDim N, G, GCI, GCO;
  std::vector<TensorDim> X(2);
  std::vector<TensorDim> KD(2);
  TensorIndex n, g, gci, gco;
  std::vector<TensorIndex> x(2);
  std::vector<TensorIndex> k(2);
  I.bind_dims(N, X[0], X[1], G, GCI);
  K.bind_dims(KD[0], KD[1], G, GCI, GCO);
  // Compute output spatial dimensions
  std::vector<TensorDim> Y(2);
  for (size_t i = 0; i < Y.size(); ++i) {
    Y[i] = (X[i] + s[i] - 1) / s[i];
  }
  // Compute the effective kernel size after dilation
  std::vector<TensorDim> EK(2);
  for (size_t i = 0; i < EK.size(); ++i) {
    EK[i] = d[i] * (KD[i] - 1) + 1;
  }
  // Compute the padding offset
  std::vector<TensorDim> P(2);
  for (size_t i = 0; i < P.size(); ++i) {
    P[i] = ((Y[i] - 1) * s[i] + EK[i] - X[i]) / 2;
  }
  // Specify the output size
  auto O = TensorOutput(N, Y[0], Y[1], G, GCO);
  // Compute the convolution
  O(n, x[0], x[1], g, gco) +=
      I(n, s[0] * x[0] + d[0] * k[0] - P[0], s[1] * x[1] + d[1] * k[1] - P[1], g, gci) * K(k[0], k[1], g, gci, gco);
  return O;
}
// complex_conv_end
TEST(CppEdsl, SumOveAxis) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  Program program("sum_over_axis", {SumOveAxis(I)});
}

TEST(CppEdsl, MaxOveAxis) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  Program program("max_over_axis", {MaxOverAxis(I)});
}

TEST(CppEdsl, MatMul) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = Placeholder(DType::UINT64, {3, 3});
  Program program("mat_mul", {MatMul(A, B)});
}

TEST(CppEdsl, GlobalMin) {
  auto I = Placeholder(DType::FLOAT32, {10, 10, 10}, "I");
  Program program("global_min", {GlobalMin(I)});
}

TEST(CppEdsl, Avg) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  Program program("avg", {Avg(I)});
}

TEST(CppEdsl, AvgStages) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  Program program("avg_stages", {AvgStages(I)});
}

TEST(CppEdsl, AvgMerge) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  Program program("avg_merge", {AvgMerge(I)});
}

TEST(CppEdsl, MaxPool1D) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  Program program("max_pool_1d", {MaxPool1D(I)});
}

TEST(CppEdsl, MaxPool1DOdd) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  Program program("max_poo_1d_odd", {MaxPool1DOdd(I)});
}

TEST(CppEdsl, Skip) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  Program program("skip", {Skip(I)});
}

TEST(CppEdsl, CumSum) {
  auto I = Placeholder(DType::FLOAT32, {10}, "I");
  Program program("cumsum", {CumSum(I)});
}

TEST(CppEdsl, Conv1D) {
  auto I = Placeholder(DType::UINT64, {1, 244, 3});
  auto K = Placeholder(DType::UINT64, {3, 3, 1});
  Program program("conv_1d", {Conv1D(I, K)});
}

TEST(CppEdsl, Conv2DDilated) {
  auto I = Placeholder(DType::UINT64, {1, 244, 244, 1});
  auto K = Placeholder(DType::UINT64, {3, 3, 1, 32});
  Program program("conv_2d_dilated", {Conv2DDilated(I, K)});
}

TEST(CppEdsl, ComplexConv2d) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3, 3});
  auto K = Placeholder(DType::FLOAT32, {3, 3, 3, 3, 32});
  auto O = ComplexConv2D(I, K, {2, 2}, {3, 3});
  Program program("complex_conv_2d", {O});
}
}  // namespace plaidml::edsl
