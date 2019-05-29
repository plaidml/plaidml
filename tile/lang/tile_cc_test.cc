// Copyright 2019 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "base/util/logging.h"
#include "tile/lang/tile_cc.h"

using ::testing::Eq;

namespace vertexai {
namespace tile {
namespace lang {
namespace {

Tensor Dot(const Tensor& X, const Tensor& Y) {
  TensorDim I, J, K;
  X.match_dims(I, K);
  Y.match_dims(K, J);
  auto R = TensorOutput(I, J);
  for (auto i : TensorIndex()) {
    for (auto j : TensorIndex()) {
      for (auto k : TensorIndex()) {
        R(i, j) += X(i, k) * Y(k, j);
      }
    }
  }
  return R;
}

Tensor Relu(const Tensor& I) { return select(I < 0.0, Tensor{0.0}, I); }

Tensor Softmax(const Tensor& X) {
  TensorDim I, J;
  X.match_dims(I, J);
  auto M = TensorOutput(I, 1);
  for (auto i : TensorIndex()) {
    for (auto j : TensorIndex()) {
      M(i, 0) >= X(i, j);
    }
  }
  auto E = exp(X - M);
  auto N = TensorOutput(I, 1);
  for (auto i : TensorIndex()) {
    for (auto j : TensorIndex()) {
      N(i, 0) += E(i, j);
    }
  }
  return E / N;
}

TEST(TileCC, MnistMlp) {
  // model.add(Dense(512, activation='relu', input_shape=(784,)))
  Tensor input(tile::SimpleShape(tile::DataType::FLOAT32, {1, 784}));
  Tensor kernel1(tile::SimpleShape(tile::DataType::FLOAT32, {784, 512}));
  Tensor bias1(tile::SimpleShape(tile::DataType::FLOAT32, {512}));
  auto dense1 = Relu(Dot(input, kernel1) + bias1);
  // model.add(Dense(512, activation='relu'))
  Tensor kernel2(tile::SimpleShape(tile::DataType::FLOAT32, {512, 512}));
  Tensor bias2(tile::SimpleShape(tile::DataType::FLOAT32, {512}));
  auto dense2 = Relu(Dot(dense1, kernel2) + bias2);
  // model.add(Dense(10, activation='softmax'))
  Tensor kernel3(tile::SimpleShape(tile::DataType::FLOAT32, {512, 10}));
  Tensor bias3(tile::SimpleShape(tile::DataType::FLOAT32, {10}));
  auto dense3 = Softmax(Dot(dense2, kernel3) + bias3);
  auto program = to_string(Evaluate("mnist_mlp", {dense3}).program);
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  _X0[_X0_0, _X0_1],
  _X1[_X1_0, _X1_1],
  _X3[_X3_0],
  _X9[_X9_0, _X9_1],
  _X11[_X11_0],
  _X17[_X17_0, _X17_1],
  _X19[_X19_0]
) -> (
  _X25
) {
  _X2[x0, x2 : 1, 512] = +(_X0[x0, x1] * _X1[x1, x2]);
  _X4 = add(_X2, _X3);
  _X5 = 0.000000;
  _X6 = cmp_lt(_X4, _X5);
  _X7 = 0.000000;
  _X8 = cond(_X6, _X7, _X4);
  _X10[x0, x2 : 1, 512] = +(_X8[x0, x1] * _X9[x1, x2]);
  _X12 = add(_X10, _X11);
  _X13 = 0.000000;
  _X14 = cmp_lt(_X12, _X13);
  _X15 = 0.000000;
  _X16 = cond(_X14, _X15, _X12);
  _X18[x0, x2 : 1, 10] = +(_X16[x0, x1] * _X17[x1, x2]);
  _X20 = add(_X18, _X19);
  _X21[x0, 0 : 1, 1] = >(_X20[x0, x1]);
  _X22 = sub(_X20, _X21);
  _X23 = exp(_X22);
  _X24[x0, 0 : 1, 1] = +(_X23[x0, x1]);
  _X25 = div(_X23, _X24);
}
)"));
}

Tensor Convolution2(const Tensor& I, const Tensor& K) {
  TensorDim CI, CO, K0, K1, N, X0, X1;
  I.match_dims(N, X0, X1, CI);
  K.match_dims(K0, K1, CI, CO);
  auto R = TensorOutput(N, X0 - (K0 - 1), X1 - (K1 - 1), CO);
  for (auto n : TensorIndex()) {
    for (auto x0 : TensorIndex()) {
      for (auto x1 : TensorIndex()) {
        for (auto co : TensorIndex()) {
          for (auto ci : TensorIndex()) {
            for (auto k0 : TensorIndex()) {
              for (auto k1 : TensorIndex()) {
                R(n, x0, x1, co) += I(n, x0 + k0 - (K0 / 2), x1 + k1 - (K1 / 2), ci) * K(k0, k1, ci, co);
              }
            }
          }
        }
      }
    }
  }
  return R;
}

Tensor MaxPooling2(const Tensor& I) {
  TensorDim N, X0, X1, C;
  I.match_dims(N, X0, X1, C);
  auto R = TensorOutput(N, (X0 + 1) / 2, (X1 + 1) / 2, C);
  for (auto n : TensorIndex()) {
    for (auto x0 : TensorIndex()) {
      for (auto x1 : TensorIndex()) {
        for (auto i : TensorIndex()) {
          for (auto j : TensorIndex()) {
            for (auto c : TensorIndex()) {
              if (i < 2 && j < 2) {
                R(n, x0, x1, c) >= I(n, 2 * x0 + i, 2 * x1 + j, c);
              }
            }
          }
        }
      }
    }
  }
  return R;
}

Tensor Flatten(const Tensor& X) {
  size_t product = 1;
  auto X_shape = X.shape();
  for (size_t i = 1; i < X_shape.dims.size() - 1; i++) {
    product *= X_shape.dims[i].size;
  }
  auto shape = tile::SimpleShape(X_shape.type, {1, product});
  return reshape(X, shape);
}

TEST(TileCC, MnistCnn) {
  // model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
  Tensor input(tile::SimpleShape(tile::DataType::FLOAT32, {1, 224, 224, 1}));
  Tensor kernel1(tile::SimpleShape(tile::DataType::FLOAT32, {3, 3, 1, 32}));
  Tensor bias1(tile::SimpleShape(tile::DataType::FLOAT32, {32}));
  auto conv1 = Relu(Convolution2(input, kernel1) + bias1);
  // model.add(Conv2D(64, (3, 3), activation='relu'))
  Tensor kernel2(tile::SimpleShape(tile::DataType::FLOAT32, {3, 3, 32, 64}));
  Tensor bias2(tile::SimpleShape(tile::DataType::FLOAT32, {64}));
  auto conv2 = Relu(Convolution2(conv1, kernel2) + bias2);
  // model.add(MaxPooling2D(pool_size=(2, 2)))
  auto pool1 = MaxPooling2(conv2);
  // model.add(Flatten())
  auto flat = Flatten(pool1);
  EXPECT_THAT(flat.shape(), Eq(tile::SimpleShape(tile::DataType::FLOAT32, {1, 12100})));
  // model.add(Dense(128, activation='relu'))
  Tensor kernel3(tile::SimpleShape(tile::DataType::FLOAT32, {12100, 128}));
  Tensor bias3(tile::SimpleShape(tile::DataType::FLOAT32, {128}));
  auto dense1 = Relu(Dot(flat, kernel3) + bias3);
  const size_t kNumClasses = 100;
  // model.add(Dense(num_classes, activation='softmax'))
  Tensor kernel4(tile::SimpleShape(tile::DataType::FLOAT32, {128, kNumClasses}));
  Tensor bias4(tile::SimpleShape(tile::DataType::FLOAT32, {kNumClasses}));
  auto dense2 = Softmax(Dot(dense1, kernel4) + bias4);
  auto program = to_string(Evaluate("mnist_cnn", {dense2}).program);
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  _X0[_X0_0, _X0_1, _X0_2, _X0_3],
  _X1[_X1_0, _X1_1, _X1_2, _X1_3],
  _X3[_X3_0],
  _X9[_X9_0, _X9_1, _X9_2, _X9_3],
  _X11[_X11_0],
  _X21[_X21_0, _X21_1],
  _X23[_X23_0],
  _X29[_X29_0, _X29_1],
  _X31[_X31_0]
) -> (
  _X37
) {
  _X2[x0, x1, x3, x6 : 1, 222, 222, 32] = +(_X0[x0, -1 + x1 + x2, -1 + x3 + x4, x5] * _X1[x2, x4, x5, x6]);
  _X4 = add(_X2, _X3);
  _X5 = 0.000000;
  _X6 = cmp_lt(_X4, _X5);
  _X7 = 0.000000;
  _X8 = cond(_X6, _X7, _X4);
  _X10[x0, x1, x3, x6 : 1, 220, 220, 64] = +(_X8[x0, -1 + x1 + x2, -1 + x3 + x4, x5] * _X9[x2, x4, x5, x6]);
  _X12 = add(_X10, _X11);
  _X13 = 0.000000;
  _X14 = cmp_lt(_X12, _X13);
  _X15 = 0.000000;
  _X16 = cond(_X14, _X15, _X12);
  _X17[x0, x1, x3, x5 : 1, 110, 110, 64] = >(_X16[x0, 2*x1 + x2, 2*x3 + x4, x5]), x2 < 2, x4 < 2;
  _X18 = 1;
  _X19 = 12100;
  _X20 = reshape(_X17, _X18, _X19);
  _X22[x0, x2 : 1, 128] = +(_X20[x0, x1] * _X21[x1, x2]);
  _X24 = add(_X22, _X23);
  _X25 = 0.000000;
  _X26 = cmp_lt(_X24, _X25);
  _X27 = 0.000000;
  _X28 = cond(_X26, _X27, _X24);
  _X30[x0, x2 : 1, 100] = +(_X28[x0, x1] * _X29[x1, x2]);
  _X32 = add(_X30, _X31);
  _X33[x0, 0 : 1, 1] = >(_X32[x0, x1]);
  _X34 = sub(_X32, _X33);
  _X35 = exp(_X34);
  _X36[x0, 0 : 1, 1] = +(_X35[x0, x1]);
  _X37 = div(_X35, _X36);
}
)"));
}

Tensor Normalize(const Tensor& X) {
  auto XSqr = X * X;
  Tensor X_MS;
  {
    std::vector<TensorIndex> idxs(X.shape().dims.size());
    X_MS() += XSqr(idxs);
  }
  return sqrt(X_MS);
}

std::tuple<Tensor, Tensor> LarsMomentum(const Tensor& X,           //
                                        const Tensor& Grad,        //
                                        const Tensor& Veloc,       //
                                        const Tensor& LR,          //
                                        double lars_coeff,         //
                                        double lars_weight_decay,  //
                                        double momentum) {
  auto XNorm = Normalize(X);
  auto GradNorm = Normalize(Grad);
  auto LocLR = LR * lars_coeff * XNorm / (GradNorm + lars_weight_decay * XNorm);
  auto NewVeloc = momentum * Veloc + LocLR * (Grad + lars_weight_decay * X);
  return std::make_tuple(X - NewVeloc, NewVeloc);
}

TEST(TileCC, LarsMomentum4d) {
  auto X_shape = tile::SimpleShape(tile::DataType::FLOAT32, {4, 7, 3, 9});
  auto LR_shape = tile::SimpleShape(tile::DataType::FLOAT32, {});
  Tensor X(X_shape);
  Tensor Grad(X_shape);
  Tensor Veloc(X_shape);
  Tensor LR(LR_shape);
  auto R = LarsMomentum(X, Grad, Veloc, LR, 1. / 1024., 1. / 2048., 1. / 8.);
  auto program = to_string(Evaluate("lars_momentum4d", {std::get<0>(R), std::get<1>(R)}).program);
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  _X1[_X1_0, _X1_1, _X1_2, _X1_3],
  _X3[],
  _X6[_X6_0, _X6_1, _X6_2, _X6_3],
  _X11[_X11_0, _X11_1, _X11_2, _X11_3]
) -> (
  _X24,
  _X23
) {
  _X0 = 0.125000;
  _X2 = mul(_X0, _X1);
  _X4 = 0.000977;
  _X5 = mul(_X3, _X4);
  _X7 = mul(_X6, _X6);
  _X8[] = +(_X7[x0, x1, x2, x3]);
  _X9 = sqrt(_X8);
  _X10 = mul(_X5, _X9);
  _X12 = mul(_X11, _X11);
  _X13[] = +(_X12[x0, x1, x2, x3]);
  _X14 = sqrt(_X13);
  _X15 = 0.000488;
  _X16 = mul(_X15, _X9);
  _X17 = add(_X14, _X16);
  _X18 = div(_X10, _X17);
  _X19 = 0.000488;
  _X20 = mul(_X19, _X6);
  _X21 = add(_X11, _X20);
  _X22 = mul(_X18, _X21);
  _X23 = add(_X2, _X22);
  _X24 = sub(_X6, _X23);
}
)"));
}

TEST(TileCC, RepeatElements) {
  Tensor I(tile::SimpleShape(tile::DataType::FLOAT32, {10, 10, 10}));
  TensorDim N0, N1, N2;
  I.match_dims(N0, N1, N2);
  auto O = TensorOutput(N0, 3 * N1, N2);
  for (auto n0 : TensorIndex()) {
    for (auto n1 : TensorIndex()) {
      for (auto n2 : TensorIndex()) {
        for (auto k : TensorIndex()) {
          if (k < 3) {
            O(n0, 3 * n1 + k, n2) = I(n0, n1, n2);
            O.no_defract();
          }
        }
      }
    }
  }
  auto program = to_string(Evaluate("repeat_elts", {O}).program);
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  _X0[_X0_0, _X0_1, _X0_2]
) -> (
  _X1
) {
  _X1[x0, 3*x1 + x3, x2 : 10, 30, 10] = =(_X0[x0, x1, x2]), x3 < 3 no_defract;
}
)"));
}

TEST(TileCC, UseDefault) {
  Tensor P(tile::SimpleShape(tile::DataType::FLOAT32, {1, 7, 10, 10}));
  Tensor I(tile::SimpleShape(tile::DataType::FLOAT32, {1, 10, 10}));
  TensorDim B, N1, N2;
  I.match_dims(B, N1, N2);
  auto O = TensorOutput(B, 7, N1, N2);
  for (auto b : TensorIndex()) {
    for (auto i1 : TensorIndex()) {
      for (auto i2 : TensorIndex()) {
        O(b, 3, i1, i2) = I(b, i1, i2);
        O.use_default(P);
      }
    }
  }
  auto program = to_string(Evaluate("use_default", {O}).program);
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  _X0[_X0_0, _X0_1, _X0_2, _X0_3],
  _X1[_X1_0, _X1_1, _X1_2]
) -> (
  _X2
) {
  _X2[x0, 3, x1, x2 : 1, 7, 10, 10] = =(_X1[x0, x1, x2]) default _X0;
}
)"));
}

Tensor ArgMax(const Tensor& I) {
  TensorDim X0, X1, X2;
  I.match_dims(X0, X1, X2);
  auto Max = TensorOutput(X0, X2);
  for (const auto x0 : TensorIndex()) {
    for (const auto x1 : TensorIndex()) {
      for (const auto x2 : TensorIndex()) {
        Max(x0, x2) >= I(x0, x1, x2);
      }
    }
  }
  Tensor One(tile::SimpleShape(tile::DataType::FLOAT32, {}));
  auto T = TensorOutput(X1);
  for (const auto x1 : TensorIndex()) {
    T(x1) = One();
  }
  Tensor IX = index(T, 0);
  auto O = TensorOutput(X0, X2);
  for (const auto x0 : TensorIndex()) {
    for (const auto x1 : TensorIndex()) {
      for (const auto x2 : TensorIndex()) {
        O(x0, x2) >= cond(I(x0, x1, x2), Max(x0, x2), IX(x1));
      }
    }
  }
  return as_uint(O, 32);
}

TEST(TileCC, ArgMax) {
  Tensor I(tile::SimpleShape(tile::DataType::FLOAT32, {1, 10, 10}));
  auto X = ArgMax(I);
  auto program = to_string(Evaluate("arg_max", {X}).program);
  IVLOG(1, program);
  EXPECT_THAT(X.shape(), Eq(tile::SimpleShape(tile::DataType::UINT32, {1, 10})));
  EXPECT_THAT(program, Eq(R"(function (
  _X0[_X0_0, _X0_1, _X0_2],
  _X2[]
) -> (
  _X8
) {
  _X1[x0, x2 : 1, 10] = >(_X0[x0, x1, x2]);
  _X3[x0 : 10] = =(_X2[]);
  _X4 = 0;
  _X5 = index(_X3, _X4);
  _X6[x0, x2 : 1, 10] = >(_X0[x0, x1, x2] == _X1[x0, x2] ? _X5[x1]);
  _X7 = 32;
  _X8 = as_uint(_X6, _X7);
}
)"));
}

Tensor Winograd(const Tensor& I, const Tensor& K, const Tensor& A, const Tensor& B, const Tensor& G) {
  TensorDim N, S, X, Y, CI, CO, BI, BO;
  I.match_dims(N, X, Y, CI);
  K.match_dims(S, S, CI, CO);
  A.match_dims(BI, BO);
  B.match_dims(BI, BI);
  G.match_dims(BI, S);
  auto XO = (X - S + 1) / 1;
  auto YO = (Y - S + 1) / 1;
  auto XB = (XO + BO - 1) / BO;
  auto YB = (YO + BO - 1) / BO;
  auto XP = 0, YP = 0;
  // assert(BI - CI + 1 == BO);
  auto U1 = TensorOutput(BI, S, CI, CO);
  auto U = TensorOutput(BI, BI, CI, CO);
  auto V1 = TensorOutput(N, BI, BI, XB, YB, CI);
  auto V = TensorOutput(N, BI, BI, XB, YB, CI);
  auto M = TensorOutput(N, BI, BI, XB, YB, CO);
  auto O1 = TensorOutput(N, BO, BI, XB, YB, CO);
  auto O = TensorOutput(N, XO, YO, CO);
  TensorIndex n, i, j, k, x, y, ci, co;
  U1(i, j, ci, co) += G(i, k) * K(k, j, ci, co);
  U(i, j, ci, co) += U1(i, k, ci, co) * G(j, k);
  V1(n, i, j, x, y, ci) += B(k, i) * I(n, BO * x + k - XP, BO * y + j - YP, ci);
  V(n, i, j, x, y, ci) += V1(n, i, k, x, y, ci) * B(k, j);
  M(n, i, j, x, y, co) += V(n, i, j, x, y, ci) * U(i, j, ci, co);
  O1(n, i, j, x, y, co) += A(k, i) * M(n, k, j, x, y, co);
  O(n, BO * x + i, BO * y + j, co) += O1(n, i, k, x, y, co) * A(k, j);
  O.no_defract();
  return O;
}

TEST(TileCC, Winograd) {
  const size_t N = 1, X = 224, Y = 224, CI = 3, S = 3, CO = 32, BI = 32, BO = BI - CI + 1;
  Tensor I(tile::SimpleShape(tile::DataType::FLOAT32, {N, X, Y, CI}));
  Tensor K(tile::SimpleShape(tile::DataType::FLOAT32, {S, S, CI, CO}));
  Tensor A(tile::SimpleShape(tile::DataType::FLOAT32, {BI, BO}));
  Tensor B(tile::SimpleShape(tile::DataType::FLOAT32, {BI, BI}));
  Tensor G(tile::SimpleShape(tile::DataType::FLOAT32, {BI, S}));
  auto W = Winograd(I, K, A, B, G);
  auto program = to_string(Evaluate("winograd", {W}).program);
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  _X0[_X0_0, _X0_1],
  _X1[_X1_0, _X1_1],
  _X2[_X2_0, _X2_1, _X2_2, _X2_3],
  _X5[_X5_0, _X5_1],
  _X6[_X6_0, _X6_1, _X6_2, _X6_3]
) -> (
  _X11
) {
  _X3[x2, x1, x5, x3, x4, x6 : 1, 32, 32, 8, 8, 3] = +(_X1[x0, x1] * _X2[x2, x0 + 30*x3, 30*x4 + x5, x6]);
  _X4[x0, x1, x6, x3, x4, x5 : 1, 32, 32, 8, 8, 3] = +(_X3[x0, x1, x2, x3, x4, x5] * _X1[x2, x6]);
  _X7[x0, x2, x3, x4 : 32, 3, 3, 32] = +(_X5[x0, x1] * _X6[x1, x2, x3, x4]);
  _X8[x0, x4, x2, x3 : 32, 32, 3, 32] = +(_X7[x0, x1, x2, x3] * _X5[x4, x1]);
  _X9[x0, x1, x2, x3, x4, x6 : 1, 32, 32, 8, 8, 32] = +(_X4[x0, x1, x2, x3, x4, x5] * _X8[x1, x2, x5, x6]);
  _X10[x2, x1, x3, x4, x5, x6 : 1, 30, 32, 8, 8, 32] = +(_X0[x0, x1] * _X9[x2, x0, x3, x4, x5, x6]);
  _X11[x0, x1 + 30*x3, 30*x4 + x6, x5 : 1, 222, 222, 32] = +(_X10[x0, x1, x2, x3, x4, x5] * _X0[x2, x6]) no_defract;
}
)"));
}

TEST(TileCC, UniqueNames) {
  Tensor A("A");
  Tensor B("B");
  Tensor C0("C");
  Tensor C1("C");
  auto program = to_string(Evaluate("unique_names", {A + B + C0 + C1}).program);
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  A[],
  B[],
  C[],
  C0[]
) -> (
  _X2
) {
  _X0 = add(A, B);
  _X1 = add(_X0, C);
  _X2 = add(_X1, C0);
}
)"));
}

TEST(TileCC, GlobalMin) {
  Tensor I("I", tile::SimpleShape(tile::DataType::FLOAT32, {10, 10, 10}));
  TensorIndex i, j, k;
  auto O_Neg = TensorOutput();
  auto Neg = -I;
  O_Neg() >= Neg(i, j, k);
  auto O = -O_Neg;
  auto program = to_string(Evaluate("global_min", {O}).program);
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2]
) -> (
  _X2
) {
  _X0 = neg(I);
  _X1[] = >(_X0[x0, x1, x2]);
  _X2 = neg(_X1);
}
)"));
}

TEST(TileCC, CumSum) {
  Tensor I("I", tile::SimpleShape(tile::DataType::FLOAT32, {10}));
  TensorDim N;
  TensorIndex i, k;
  I.match_dims(N);
  auto O = TensorOutput(N);
  if (i - k < N) {
    O(i) += I(k);
  }
  auto program = to_string(Evaluate("csum", {O}).program);
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0]
) -> (
  _X0
) {
  _X0[x1 : 10] = +(I[x0]), -x0 + x1 < 10;
}
)"));
}

Tensor ComplexConv2d(const Tensor& I,               //
                     const Tensor& K,               //
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
  I.match_dims(N, X[0], X[1], G, GCI);
  K.match_dims(KX[0], KX[1], G, GCI, GCO);
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
  // Specify the output size
  auto O = TensorOutput(N, Y[0], Y[1], G, GCO);
  // Compute the convolution
  O(n, x[0], x[1], g, gco) +=
      I(n, s[0] * x[0] + d[0] * k[0] - P[0], s[1] * x[1] + d[1] * k[1] - P[1], g, gci) * K(k[0], k[1], g, gci, gco);
  return O;
}

TEST(TileCC, ComplexConv2d) {
  Tensor I(tile::SimpleShape(tile::DataType::FLOAT32, {1, 224, 224, 3, 3}));
  Tensor K(tile::SimpleShape(tile::DataType::FLOAT32, {3, 3, 3, 3, 32}));
  auto O = ComplexConv2d(I, K, {2, 2}, {3, 3});
  auto program = to_string(Evaluate("complex_conv_2d", {O}).program);
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  _X0[_X0_0, _X0_1, _X0_2, _X0_3, _X0_4],
  _X1[_X1_0, _X1_1, _X1_2, _X1_3, _X1_4]
) -> (
  _X2
) {
  _X2[x0, x1, x3, x5, x7 : 1, 112, 112, 3, 32] = +(_X0[x0, -2 + 2*x1 + 3*x2, -2 + 2*x3 + 3*x4, x5, x6] * _X1[x2, x4, x5, x6, x7]);
}
)"));
}

}  // namespace
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
