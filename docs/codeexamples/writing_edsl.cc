
// Copyright 2019 Intel Corporation.
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

Tensor SumOveAxis(const Tensor& I) {
  TensorDim M, N;
  TensorIndex m, n;
  I.bind_dims(M, N);
  auto O = TensorOutput(N);
  O(n) += I(m, n);  // contraction
  return O;
}

Tensor MaxOverAxis(const Tensor& I) {
  TensorDim M, N;
  TensorIndex m, n;
  I.bind_dims(M, N);
  auto O = TensorOutput(N);
  O(n) >= I(m, n);
  return O;
}

Tensor MatMul(const Tensor& A, const Tensor& B) {
  TensorDim I, J, K;
  TensorIndex i, j, k;
  A.bind_dims(I, K);
  B.bind_dims(K, J);
  auto C = TensorOutput(I, J);
  C(i, j) += A(i, k) * B(k, j);
  return C;
}

Tensor GlobalMin(const Tensor& I) {
  TensorIndex i, j, k;
  auto Neg = -I;
  auto O_Neg = TensorOutput();
  O_Neg() >= Neg(i, j, k);
  auto O = -O_Neg;
  return O;
}

Tensor Avg(const Tensor& I) {
  TensorDim X, Y;
  TensorIndex x, y;
  I.bind_dims(X, Y);
  auto Sum = TensorOutput();
  Sum(y) += I(x, y);
  return Sum / X;
}

Tensor AvgStages(const Tensor& I) {
  TensorDim X, Y;
  TensorIndex x, y;
  I.bind_dims(X, Y);
  auto Sum = TensorOutput();
  Sum() += I(x, y);
  auto PartialMean = Sum / X;
  return PartialMean / Y;
}

Tensor AvgMerge(const Tensor& I) {
  TensorDim X, Y;
  TensorIndex x, y;
  I.bind_dims(X, Y);
  auto Sum = TensorOutput();
  Sum() += I(x, y);
  return Sum / (X * Y);
}

void ForLoopMaxPool1D() {
  int N = 10;
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
}

Tensor WrongMaxPool1D(const Tensor& I) {
  TensorDim N;
  TensorIndex i, j;
  I.bind_dims(N);
  auto O = TensorOutput(N / 2);
  O(i) >= I(2 * i + j);
  return O;
}

Tensor MaxPool1D(const Tensor& I) {
  TensorDim N;
  TensorIndex i, j;
  I.bind_dims(N);
  auto O = TensorOutput(N / 2);
  O(i) >= I(2 * i + j);
  O.add_constraint(j < 2);
  return O;
}

Tensor MaxPool1DOdd(const Tensor& I) {
  TensorDim N;
  TensorIndex i, j;
  I.bind_dims(N);
  auto O = TensorOutput((N + 1) / 2);
  O(i) >= I(2 * i + j);
  O.add_constraint(j < 2);
  return O;
}

Tensor Skip(const Tensor& I) {
  TensorDim M, N;
  TensorIndex i, j;
  I.bind_dims(M, N);
  auto O = TensorOutput(N);
  O(2 * i) += I(2 * i, j);
  return O;
}

Tensor CumSum(const Tensor& I) {
  TensorDim N;
  TensorIndex i, k;
  I.bind_dims(N);
  auto O = TensorOutput(N);
  O(i) += I(k);
  O.add_constraint(i - k < N);
  return O;
}

Tensor Conv1D(const Tensor& I, const Tensor& K) {
  TensorDim N, X, KX, CI, CO;
  TensorIndex n, x, k, ci, co;
  I.bind_dims(N, X, CI);
  K.bind_dims(KX, CI, CO);
  auto O = TensorOutput(N, X - KX + 1, CO);
  O(n, x, co) += I(n, x + k, ci) * K(k, ci, co);
  return O;
}

Tensor Conv2DDilated(const Tensor& I, const Tensor& K) {
  TensorDim N, X, Y, KX, KY, CI, CO;
  TensorIndex n, x, y, kx, ky, ci, co;
  I.bind_dims(N, X, Y, CI);
  K.bind_dims(KX, KY, CI, CO);
  auto O = TensorOutput(N, X - 2 * (KX - 1), Y - 3 * (KY - 1), CO);
  O(n, x, y, co) += I(n, x + 2 * kx, y + 3 * ky, ci) * K(kx, ky, ci, co);
  return O;
}

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

TEST(CppEdsl, SumOveAxis) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  Program program("sum_over_axis", {SumOveAxis(I)});
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>


!f32 = type tensor<!eltwise.f32>
module {
  func @sum_over_axis(%arg0: tensor<3x3x!eltwise.u64>) -> tensor<3x!eltwise.u64> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %0 = tile.cion add, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : !f32, tensor<3x3x!eltwise.u64> -> tensor<3x!eltwise.u64>
    return %0 : tensor<3x!eltwise.u64>
  }
}
)#"));
}

TEST(CppEdsl, MaxOveAxis) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  Program program("max_over_axis", {MaxOverAxis(I)});
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>


!f32 = type tensor<!eltwise.f32>
module {
  func @max_over_axis(%arg0: tensor<3x3x!eltwise.u64>) -> tensor<3x!eltwise.u64> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %0 = tile.cion max, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : !f32, tensor<3x3x!eltwise.u64> -> tensor<3x!eltwise.u64>
    return %0 : tensor<3x!eltwise.u64>
  }
}
)#"));
}

TEST(CppEdsl, MatMul) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = Placeholder(DType::UINT64, {3, 3});
  Program program("mat_mul", {MatMul(A, B)});
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>


!f32 = type tensor<!eltwise.f32>
module {
  func @mat_mul(%arg0: tensor<3x3x!eltwise.u64>, %arg1: tensor<3x3x!eltwise.u64>) -> tensor<3x3x!eltwise.u64> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %0 = tile.cion add, mul, %cst, %arg1, %arg0 {sink = #map0, srcs = [#map1, #map2]} : !f32, tensor<3x3x!eltwise.u64>, tensor<3x3x!eltwise.u64> -> tensor<3x3x!eltwise.u64>
    return %0 : tensor<3x3x!eltwise.u64>
  }
}
)#"));
}

TEST(CppEdsl, GlobalMin) {
  auto I = Placeholder(DType::FLOAT32, {10, 10, 10}, "I");
  Program program("global_min", {GlobalMin(I)});
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>


!f32 = type tensor<!eltwise.f32>
module {
  func @global_min(%arg0: tensor<10x10x10x!eltwise.f32> {tile.name = "I"}) -> !f32 {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %0 = "eltwise.neg"(%arg0) : (tensor<10x10x10x!eltwise.f32>) -> tensor<10x10x10x!eltwise.f32>
    %1 = tile.cion max, none, %cst, %0 {sink = #map0, srcs = [#map1]} : !f32, tensor<10x10x10x!eltwise.f32> -> !f32
    %2 = "eltwise.neg"(%1) : (!f32) -> !f32
    return %2 : !f32
  }
}
)#"));
#if !defined(_WIN32)
  exec::Binder(program).compile()->run();
#endif
}

TEST(CppEdsl, Avg) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  Program program("avg", {Avg(I)});
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>


!f32 = type tensor<!eltwise.f32>
!u64 = type tensor<!eltwise.u64>
module {
  func @avg(%arg0: tensor<3x3x!eltwise.u64>) -> !u64 {
    %c3 = tile.affine_const 3
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %0 = tile.cion add, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : !f32, tensor<3x3x!eltwise.u64> -> !u64
    %1 = "eltwise.div"(%0, %c3) : (!u64, index) -> !u64
    return %1 : !u64
  }
}
)#"));
}

TEST(CppEdsl, AvgStages) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  Program program("avg_stages", {AvgStages(I)});
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>


!f32 = type tensor<!eltwise.f32>
!u64 = type tensor<!eltwise.u64>
module {
  func @avg_stages(%arg0: tensor<3x3x!eltwise.u64>) -> !u64 {
    %c3 = tile.affine_const 3
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %0 = tile.cion add, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : !f32, tensor<3x3x!eltwise.u64> -> !u64
    %1 = "eltwise.div"(%0, %c3) : (!u64, index) -> !u64
    %2 = "eltwise.div"(%1, %c3) : (!u64, index) -> !u64
    return %2 : !u64
  }
}
)#"));
}

TEST(CppEdsl, AvgMerge) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  Program program("avg_merge", {AvgMerge(I)});
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>


!f32 = type tensor<!eltwise.f32>
!u64 = type tensor<!eltwise.u64>
module {
  func @avg_merge(%arg0: tensor<3x3x!eltwise.u64>) -> !u64 {
    %c9 = tile.affine_const 9
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %0 = tile.cion add, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : !f32, tensor<3x3x!eltwise.u64> -> !u64
    %1 = "eltwise.div"(%0, %c9) : (!u64, index) -> !u64
    return %1 : !u64
  }
}
)#"));
}

TEST(CppEdsl, MaxPool1D) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  Program program("max_pool_1d", {MaxPool1D(I)});
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0 * 2 + d1)>

#set0 = affine_set<(d0, d1) : (d1 >= 0, -d1 + 1 >= 0)>

!f32 = type tensor<!eltwise.f32>
module {
  func @max_pool_1d(%arg0: tensor<3x3x!eltwise.u64>) -> tensor<1x!eltwise.u64> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %0 = tile.cion max, none, %cst, %arg0 {cons = #set0, sink = #map0, srcs = [#map1]} : !f32, tensor<3x3x!eltwise.u64> -> tensor<1x!eltwise.u64>
    return %0 : tensor<1x!eltwise.u64>
  }
}
)#"));
}

TEST(CppEdsl, MaxPool1DOdd) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  Program program("max_poo_1d_odd", {MaxPool1DOdd(I)});
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0 * 2 + d1)>

#set0 = affine_set<(d0, d1) : (d1 >= 0, -d1 + 1 >= 0)>

!f32 = type tensor<!eltwise.f32>
module {
  func @max_poo_1d_odd(%arg0: tensor<3x3x!eltwise.u64>) -> tensor<2x!eltwise.u64> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %0 = tile.cion max, none, %cst, %arg0 {cons = #set0, sink = #map0, srcs = [#map1]} : !f32, tensor<3x3x!eltwise.u64> -> tensor<2x!eltwise.u64>
    return %0 : tensor<2x!eltwise.u64>
  }
}
)#"));
}

TEST(CppEdsl, Skip) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  Program program("skip", {Skip(I)});
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1) -> (d0 * 2)>
#map1 = affine_map<(d0, d1) -> (d0 * 2, d1)>


!f32 = type tensor<!eltwise.f32>
module {
  func @skip(%arg0: tensor<3x3x!eltwise.u64>) -> tensor<3x!eltwise.u64> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %0 = tile.cion add, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : !f32, tensor<3x3x!eltwise.u64> -> tensor<3x!eltwise.u64>
    return %0 : tensor<3x!eltwise.u64>
  }
}
)#"));
  exec::Binder(program).compile()->run();
}

TEST(CppEdsl, CumSum) {
  auto I = Placeholder(DType::FLOAT32, {10}, "I");
  Program program("cumsum", {CumSum(I)});
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>

#set0 = affine_set<(d0, d1) : (d0 - d1 >= 0, -d0 + d1 + 9 >= 0)>

!f32 = type tensor<!eltwise.f32>
module {
  func @cumsum(%arg0: tensor<10x!eltwise.f32> {tile.name = "I"}) -> tensor<10x!eltwise.f32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %0 = tile.cion add, none, %cst, %arg0 {cons = #set0, sink = #map0, srcs = [#map1]} : !f32, tensor<10x!eltwise.f32> -> tensor<10x!eltwise.f32>
    return %0 : tensor<10x!eltwise.f32>
  }
}
)#"));
  exec::Binder(program).compile()->run();
}

TEST(CppEdsl, Conv1D) {
  auto I = Placeholder(DType::UINT64, {1, 244, 3});
  auto K = Placeholder(DType::UINT64, {3, 3, 1});
  Program program("conv_1d", {Conv1D(I, K)});
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 + d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>


!f32 = type tensor<!eltwise.f32>
module {
  func @conv_1d(%arg0: tensor<3x3x1x!eltwise.u64>, %arg1: tensor<1x244x3x!eltwise.u64>) -> tensor<1x242x1x!eltwise.u64> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %0 = tile.cion add, mul, %cst, %arg1, %arg0 {sink = #map0, srcs = [#map1, #map2]} : !f32, tensor<1x244x3x!eltwise.u64>, tensor<3x3x1x!eltwise.u64> -> tensor<1x242x1x!eltwise.u64>
    return %0 : tensor<1x242x1x!eltwise.u64>
  }
}
)#"));
  exec::Binder(program).compile()->run();
}

TEST(CppEdsl, Conv2DDilated) {
  auto I = Placeholder(DType::UINT64, {1, 244, 244, 1});
  auto K = Placeholder(DType::UINT64, {3, 3, 1, 32});
  Program program("conv_2d_dilated", {Conv2DDilated(I, K)});
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 * 2, d2 + d5 * 3, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>


!f32 = type tensor<!eltwise.f32>
module {
  func @conv_2d_dilated(%arg0: tensor<3x3x1x32x!eltwise.u64>, %arg1: tensor<1x244x244x1x!eltwise.u64>) -> tensor<1x240x238x32x!eltwise.u64> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %0 = tile.cion add, mul, %cst, %arg1, %arg0 {sink = #map0, srcs = [#map1, #map2]} : !f32, tensor<1x244x244x1x!eltwise.u64>, tensor<3x3x1x32x!eltwise.u64> -> tensor<1x240x238x32x!eltwise.u64>
    return %0 : tensor<1x240x238x32x!eltwise.u64>
  }
}
)#"));
  exec::Binder(program).compile()->run();
}

TEST(CppEdsl, ComplexConv2d) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3, 3});
  auto K = Placeholder(DType::FLOAT32, {3, 3, 3, 3, 32});
  auto O = ComplexConv2D(I, K, {2, 2}, {3, 3});
  Program program("complex_conv_2d", {O});
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5 * 3 - 2, d2 * 2 + d6 * 3 - 2, d3, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d3, d7, d4)>


!f32 = type tensor<!eltwise.f32>
module {
  func @complex_conv_2d(%arg0: tensor<3x3x3x3x32x!eltwise.f32>, %arg1: tensor<1x224x224x3x3x!eltwise.f32>) -> tensor<1x112x112x3x32x!eltwise.f32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %0 = tile.cion add, mul, %cst, %arg1, %arg0 {sink = #map0, srcs = [#map1, #map2]} : !f32, tensor<1x224x224x3x3x!eltwise.f32>, tensor<3x3x3x3x32x!eltwise.f32> -> tensor<1x112x112x3x32x!eltwise.f32>
    return %0 : tensor<1x112x112x3x32x!eltwise.f32>
  }
}
)#"));
  exec::Binder(program).compile()->run();
}

}  // namespace plaidml::edsl
