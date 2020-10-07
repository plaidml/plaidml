// Copyright 2020 Intel Corporation.
// Note:
//    This file is being used by sphinx docs to pull in code blocks.
//    Code blocks are pulled into docs/usage/writing_edsl.rs
//    Any changes made here may upset the docs.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "plaidml/edsl/edsl.h"
#include "plaidml/testenv.h"

namespace plaidml::edsl {

class DocCppEdsl : public TestFixture {};

// sum_over_axis_start
Tensor SumOverAxis(const Tensor& I) {
  TensorDim M, N;
  TensorIndex m, n;
  I.bind_dims(M, N);
  return Contraction().outShape(N).outAccess(n).sum(I(m, n));  // contraction
}
// sum_over_axis_end

// max_over_axis_start
Tensor MaxOverAxis(const Tensor& I) {
  TensorDim M, N;
  TensorIndex m, n;
  I.bind_dims(M, N);
  return Contraction().outShape(N).outAccess(n).max(I(m, n));
}
// max_over_axis_end

// matmul_start
Tensor MatMul(const Tensor& A, const Tensor& B) {
  TensorDim I, J, K;
  TensorIndex i, j, k;
  A.bind_dims(I, K);
  B.bind_dims(K, J);
  return Contraction().outShape(I, J).outAccess(i, j).sum(A(i, k) * B(k, j));
}
// matmul_end

// global_min_start
Tensor GlobalMin(const Tensor& I) {
  TensorIndex i, j, k;
  auto Neg = -I;
  Tensor O = Contraction().max(Neg(i, j, k));
  return -O;
}
// global_min_end

// avg_start
Tensor Avg(const Tensor& I) {
  TensorDim X, Y;
  TensorIndex x, y;
  I.bind_dims(X, Y);
  Tensor Sum = Contraction().outShape(Y).outAccess(y).sum(I(x, y));
  return Sum / X;
}
// avg_end

// avg_stages_start
Tensor AvgStages(const Tensor& I) {
  TensorDim X, Y;
  TensorIndex x, y;
  I.bind_dims(X, Y);
  Tensor Sum = Contraction().sum(I(x, y));
  auto PartialMean = Sum / X;
  return PartialMean / Y;
}
// avg_stages_end

// avg_merge_start
Tensor AvgMerge(const Tensor& I) {
  TensorDim X, Y;
  TensorIndex x, y;
  I.bind_dims(X, Y);
  Tensor Sum = Contraction().sum(I(x, y));
  return Sum / (X * Y);
}
// avg_merge_end

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

Tensor ValidIndices(const Tensor& I) {
  TensorDim N;
  TensorIndex i, j;
  // valid_indices_start
  I.bind_dims(N);
  Tensor O = Contraction().outShape(N / 2).outAccess(i).max(I(2 * i + j)).add_constraint(j < 2);
  // valid_indices_end
  return O;
}

// max_pool_1d_start
Tensor MaxPool1D(const Tensor& I) {
  TensorDim N;
  TensorIndex i, j;
  I.bind_dims(N);
  return Contraction().outShape(N / 2).outAccess(i).max(I(2 * i + j)).add_constraint(j < 2);
}
// max_pool_1d_end

// max_pool_1d_odd_start
Tensor MaxPool1DOdd(const Tensor& I) {
  TensorDim N;
  TensorIndex i, j;
  I.bind_dims(N);
  return Contraction().outShape((N + 1) / 2).outAccess(i).max(I(2 * i + j)).add_constraint(j < 2);
}
// max_pool_1d_odd_end

// skip_start
Tensor Skip(const Tensor& I) {
  TensorDim M, N;
  TensorIndex i, j;
  I.bind_dims(M, N);
  return Contraction().outShape(N).outAccess(2 * i).sum(I(2 * i, j));
}
// skip_end

// cumsum_start
Tensor CumSum(const Tensor& I) {
  TensorDim N;
  TensorIndex i, k;
  I.bind_dims(N);
  return Contraction().outShape(N).outAccess(i).sum(I(k)).add_constraint(i - k < N);
}
// cumsum_end

// conv_1d_start
Tensor Conv1D(const Tensor& I, const Tensor& K) {
  TensorDim N, X, KX, CI, CO;
  TensorIndex n, x, k, ci, co;
  I.bind_dims(N, X, CI);
  K.bind_dims(KX, CI, CO);
  return Contraction().outShape(N, X - KX + 1, CO).outAccess(n, x, co).sum(I(n, x + k, ci) * K(k, ci, co));
}
// conv_1d_end

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
  // Compute the convolution
  return Contraction()
      .outShape(N, Y[0], Y[1], G, GCO)
      .outAccess(n, x[0], x[1], g, gco)
      .sum(I(n, s[0] * x[0] + d[0] * k[0] - P[0], s[1] * x[1] + d[1] * k[1] - P[1], g, gci) *
           K(k[0], k[1], g, gci, gco));
}
// complex_conv_end

TEST_F(DocCppEdsl, SumOveAxis) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  runProgram(makeProgram("sum_over_axis", {I}, {SumOverAxis(I)}));
}

TEST_F(DocCppEdsl, MaxOverAxis) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  runProgram(makeProgram("max_over_axis", {I}, {MaxOverAxis(I)}));
}

TEST_F(DocCppEdsl, MatMul) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = Placeholder(DType::UINT64, {3, 3});
  runProgram(makeProgram("mat_mul", {A, B}, {MatMul(A, B)}));
}

TEST_F(DocCppEdsl, GlobalMin) {
  auto I = Placeholder(DType::FLOAT32, {10, 10, 10}, "I");
  runProgram(makeProgram("global_min", {I}, {GlobalMin(I)}));
}

TEST_F(DocCppEdsl, Avg) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  runProgram(makeProgram("avg", {I}, {Avg(I)}));
}

TEST_F(DocCppEdsl, AvgStages) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  runProgram(makeProgram("avg_stages", {I}, {AvgStages(I)}));
}

TEST_F(DocCppEdsl, AvgMerge) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  runProgram(makeProgram("avg_merge", {I}, {AvgMerge(I)}));
}

TEST_F(DocCppEdsl, WrongMaxPool1D) {
  auto I = Placeholder(DType::UINT64, {3});
  makeProgram("wrong_max_pool_1d", {I}, {WrongMaxPool1D(I)});
}

TEST_F(DocCppEdsl, ValidIndices) {
  auto I = Placeholder(DType::UINT64, {3});
  runProgram(makeProgram("valid_indices", {I}, {ValidIndices(I)}));
}

TEST_F(DocCppEdsl, MaxPool1D) {
  auto I = Placeholder(DType::UINT64, {3});
  runProgram(makeProgram("max_pool_1d", {I}, {MaxPool1D(I)}));
}

TEST_F(DocCppEdsl, MaxPool1DOdd) {
  auto I = Placeholder(DType::UINT64, {3});
  runProgram(makeProgram("max_poo_1d_odd", {I}, {MaxPool1DOdd(I)}));
}

TEST_F(DocCppEdsl, Skip) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  runProgram(makeProgram("skip", {I}, {Skip(I)}));
}

TEST_F(DocCppEdsl, CumSum) {
  auto I = Placeholder(DType::FLOAT32, {10}, "I");
  runProgram(makeProgram("cumsum", {I}, {CumSum(I)}));
}

TEST_F(DocCppEdsl, Conv1D) {
  auto I = Placeholder(DType::UINT64, {1, 244, 3});
  auto K = Placeholder(DType::UINT64, {3, 3, 1});
  runProgram(makeProgram("conv_1d", {I, K}, {Conv1D(I, K)}));
}

TEST_F(DocCppEdsl, Conv2DDilated) {
  auto I = Placeholder(DType::UINT64, {1, 244, 244, 1});
  auto K = Placeholder(DType::UINT64, {3, 3, 1, 32});
  runProgram(makeProgram("conv_2d_dilated", {I, K}, {Conv2DDilated(I, K)}));
}

TEST_F(DocCppEdsl, ComplexConv2d) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3, 3});
  auto K = Placeholder(DType::FLOAT32, {3, 3, 3, 3, 32});
  auto O = ComplexConv2D(I, K, {2, 2}, {3, 3});
  runProgram(makeProgram("complex_conv_2d", {I, K}, {O}));
}

}  // namespace plaidml::edsl
