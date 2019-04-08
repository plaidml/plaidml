// Copyright 2019 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "base/util/logging.h"
#include "plaidml/plaidml++.h"
#include "plaidml/tile_cc.h"
#include "testing/plaidml_config.h"

using ::testing::Eq;

namespace vertexai {
namespace plaidml {
namespace tile_cc {
namespace {

Tensor Dot(const Tensor& X, const Tensor& Y) {
  Tensor R;
  for (auto x0 : Index()) {
    for (auto y1 : Index()) {
      for (auto z : Index()) {
        R({x0, y1}, {X[0], Y[1]}) += X({x0, z}) * Y({z, y1});
      }
    }
  }
  return R;
}

Tensor Relu(const Tensor& I) { return select(I < 0.0, 0.0, I); }

Tensor Softmax(const Tensor& X) {
  Tensor M;
  for (auto i : Index()) {
    for (auto j : Index()) {
      M({i, 0}, {X[0], 1}) >= X({i, j});
    }
  }
  auto E = exp(X - M);
  Tensor N;
  for (auto i : Index()) {
    for (auto j : Index()) {
      N({i, 0}, {X[0], 1}) += E({i, j});
    }
  }
  return E / N;
}

TEST(TileCC, MnistMlp) {
  Tensor input(tile::SimpleShape(tile::DataType::FLOAT32, {1, 512}));
  Tensor kernel1(tile::SimpleShape(tile::DataType::FLOAT32, {784, 512}));
  Tensor bias1(tile::SimpleShape(tile::DataType::FLOAT32, {512}));
  // model.add(Dense(512, activation='relu', input_shape=(784,)))
  auto dense1 = Relu(Dot(input, kernel1) + bias1);
  Tensor kernel2(tile::SimpleShape(tile::DataType::FLOAT32, {784, 512}));
  Tensor bias2(tile::SimpleShape(tile::DataType::FLOAT32, {512}));
  // model.add(Dense(512, activation='relu'))
  auto dense2 = Relu(Dot(dense1, kernel2) + bias2);
  // model.add(Dense(10, activation='softmax'))
  Tensor kernel3(tile::SimpleShape(tile::DataType::FLOAT32, {784, 10}));
  Tensor bias3(tile::SimpleShape(tile::DataType::FLOAT32, {10}));
  auto dense3 = Softmax(Dot(dense2, kernel3) + bias3);
  auto program = to_string(Evaluate({dense3}));
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  X0[X0_0, X0_1],
  X1[X1_0, X1_1],
  X3[X3_0],
  X9[X9_0, X9_1],
  X11[X11_0],
  X17[X17_0, X17_1],
  X19[X19_0]
) -> (
  X25
) {
  X2[x0, x2 : 1, 512] = +(X0[x0, x1] * X1[x1, x2]);
  X4 = add(X2, X3);
  X5 = 0.000000;
  X6 = cmp_lt(X4, X5);
  X7 = 0.000000;
  X8 = cond(X6, X7, X4);
  X10[x0, x2 : 1, 512] = +(X8[x0, x1] * X9[x1, x2]);
  X12 = add(X10, X11);
  X13 = 0.000000;
  X14 = cmp_lt(X12, X13);
  X15 = 0.000000;
  X16 = cond(X14, X15, X12);
  X18[x0, x2 : 1, 10] = +(X16[x0, x1] * X17[x1, x2]);
  X20 = add(X18, X19);
  X21[x0, 0 : 1, 1] = >(X20[x0, x1]);
  X22 = sub(X20, X21);
  X23 = exp(X22);
  X24[x0, 0 : 1, 1] = +(X23[x0, x1]);
  X25 = div(X23, X24);
}
)"));
}

Tensor Convolution2(const Tensor& I, const Tensor& K) {
  Tensor R;
  auto N = I[0], H = I[1], W = I[2];
  auto KH = K[0], KW = K[1], CO = K[3];
  auto kc0 = K.shape().dims[0].size / 2;
  auto kc1 = K.shape().dims[1].size / 2;
  for (auto n : Index()) {
    for (auto x0 : Index()) {
      for (auto x1 : Index()) {
        for (auto co : Index()) {
          for (auto ci : Index()) {
            for (auto k0 : Index()) {
              for (auto k1 : Index()) {
                R({n, x0, x1, co}, {N, H - (KH - 1), W - (KW - 1), CO}) +=
                    I({n, x0 + k0 - kc0, x1 + k1 - kc1, ci}) * K({k0, k1, ci, co});
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
  Tensor R;
  auto N = I[0], H = I[1], W = I[2], C = I[3];
  for (auto n : Index()) {
    for (auto h : Index()) {
      for (auto w : Index()) {
        for (auto i : Index()) {
          for (auto j : Index()) {
            for (auto c : Index()) {
              if (i < 2 && j < 2) {
                R({n, h, w, c}, {N, (H + 1) / 2, (W + 1) / 2, C}) >= I({n, 2 * h + i, 2 * w + j, c});
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
  for (size_t i = 1; i < X.shape().dims.size() - 1; i++) {
    product *= X.shape().dims[i].size;
  }
  auto shape = tile::SimpleShape(X.shape().type, {1, product});
  return reshape(X, shape);
}

TEST(TileCC, MnistCnn) {
  Tensor input(tile::SimpleShape(tile::DataType::FLOAT32, {1, 224, 224, 1}));
  Tensor kernel1(tile::SimpleShape(tile::DataType::FLOAT32, {3, 3, 1, 32}));
  Tensor bias1(tile::SimpleShape(tile::DataType::FLOAT32, {32}));
  // model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
  auto conv1 = Relu(Convolution2(input, kernel1) + bias1);
  Tensor kernel2(tile::SimpleShape(tile::DataType::FLOAT32, {3, 3, 32, 64}));
  Tensor bias2(tile::SimpleShape(tile::DataType::FLOAT32, {64}));
  // model.add(Conv2D(64, (3, 3), activation='relu'))
  auto conv2 = Relu(Convolution2(conv1, kernel2) + bias2);
  // model.add(MaxPooling2D(pool_size=(2, 2)))
  auto pool1 = MaxPooling2(conv2);
  // model.add(Flatten())
  auto flat = Flatten(pool1);
  EXPECT_THAT(flat.shape(), Eq(tile::SimpleShape(tile::DataType::FLOAT32, {1, 12100})));
  Tensor kernel3(tile::SimpleShape(tile::DataType::FLOAT32, {64, 128}));
  Tensor bias3(tile::SimpleShape(tile::DataType::FLOAT32, {128}));
  // model.add(Dense(128, activation='relu'))
  auto dense1 = Relu(Dot(flat, kernel3) + bias3);
  const size_t kNumClasses = 100;
  Tensor kernel4(tile::SimpleShape(tile::DataType::FLOAT32, {128, kNumClasses}));
  Tensor bias4(tile::SimpleShape(tile::DataType::FLOAT32, {kNumClasses}));
  // model.add(Dense(num_classes, activation='softmax'))
  auto dense2 = Softmax(Dot(dense1, kernel4) + bias4);
  auto program = to_string(Evaluate({dense2}));
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  X0[X0_0, X0_1, X0_2, X0_3],
  X1[X1_0, X1_1, X1_2, X1_3],
  X3[X3_0],
  X9[X9_0, X9_1, X9_2, X9_3],
  X11[X11_0],
  X21[X21_0, X21_1],
  X23[X23_0],
  X29[X29_0, X29_1],
  X31[X31_0]
) -> (
  X37
) {
  X2[x0, x1, x3, x6 : 1, 222, 222, 32] = +(X0[x0, -1 + x1 + x2, -1 + x3 + x4, x5] * X1[x2, x4, x5, x6]);
  X4 = add(X2, X3);
  X5 = 0.000000;
  X6 = cmp_lt(X4, X5);
  X7 = 0.000000;
  X8 = cond(X6, X7, X4);
  X10[x0, x1, x3, x6 : 1, 220, 220, 64] = +(X8[x0, -1 + x1 + x2, -1 + x3 + x4, x5] * X9[x2, x4, x5, x6]);
  X12 = add(X10, X11);
  X13 = 0.000000;
  X14 = cmp_lt(X12, X13);
  X15 = 0.000000;
  X16 = cond(X14, X15, X12);
  X17[x0, x1, x3, x5 : 1, 110, 110, 64] = >(X16[x0, 2*x1 + x2, 2*x3 + x4, x5]), x2 < 2, x4 < 2;
  X18 = 1;
  X19 = 12100;
  X20 = reshape(X17, X18, X19);
  X22[x0, x2 : 1, 128] = +(X20[x0, x1] * X21[x1, x2]);
  X24 = add(X22, X23);
  X25 = 0.000000;
  X26 = cmp_lt(X24, X25);
  X27 = 0.000000;
  X28 = cond(X26, X27, X24);
  X30[x0, x2 : 1, 100] = +(X28[x0, x1] * X29[x1, x2]);
  X32 = add(X30, X31);
  X33[x0, 0 : 1, 1] = >(X32[x0, x1]);
  X34 = sub(X32, X33);
  X35 = exp(X34);
  X36[x0, 0 : 1, 1] = +(X35[x0, x1]);
  X37 = div(X35, X36);
}
)"));
}

Tensor Normalize(const Tensor& X) {
  auto XSqr = X * X;
  Tensor X_MS;
  {
    std::vector<Index> idxs(X.shape().dims.size());
    X_MS({}) += XSqr(idxs);
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
  auto program = to_string(Evaluate({std::get<0>(R), std::get<1>(R)}));
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  X0[X0_0, X0_1, X0_2, X0_3],
  X1[X1_0, X1_1, X1_2, X1_3],
  X4[],
  X11[X11_0, X11_1, X11_2, X11_3]
) -> (
  X24,
  X23
) {
  X2 = 0.125000;
  X3 = mul(X1, X2);
  X5 = 0.000977;
  X6 = mul(X4, X5);
  X7 = mul(X0, X0);
  X8[] = +(X7[x0, x1, x2, x3]);
  X9 = sqrt(X8);
  X10 = mul(X6, X9);
  X12 = mul(X11, X11);
  X13[] = +(X12[x0, x1, x2, x3]);
  X14 = sqrt(X13);
  X15 = 0.000488;
  X16 = mul(X9, X15);
  X17 = add(X14, X16);
  X18 = div(X10, X17);
  X19 = 0.000488;
  X20 = mul(X0, X19);
  X21 = add(X11, X20);
  X22 = mul(X18, X21);
  X23 = add(X3, X22);
  X24 = sub(X0, X23);
}
)"));
}

TEST(TileCC, RepeatElements) {
  Tensor I(tile::SimpleShape(tile::DataType::FLOAT32, {10, 10, 10}));
  auto N0 = I[0], N1 = I[1], N2 = I[2];
  Tensor O;
  for (auto n0 : Index()) {
    for (auto n1 : Index()) {
      for (auto n2 : Index()) {
        for (auto k : Index()) {
          if (k < 3) {
            O({n0, 3 * n1 + k, n2}, {N0, 3 * N1, N2}) = I({n0, n1, n2});
            O.no_defract();
          }
        }
      }
    }
  }
  auto program = to_string(Evaluate({O}));
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  X0[X0_0, X0_1, X0_2]
) -> (
  X1
) {
  X1[x0, 3*x1 + x3, x2 : 10, 30, 10] = =(X0[x0, x1, x2]), x3 < 3 no_defract;
}
)"));
}

TEST(TileCC, UseDefault) {
  Tensor P(tile::SimpleShape(tile::DataType::FLOAT32, {1, 7, 10, 10}));
  Tensor I(tile::SimpleShape(tile::DataType::FLOAT32, {1, 10, 10}));
  auto B = I[0], N1 = I[1], N2 = I[2];
  Tensor O;
  for (auto b : Index()) {
    for (auto i1 : Index()) {
      for (auto i2 : Index()) {
        O({b, 3, i1, i2}, {B, 7, N1, N2}) = I({b, i1, i2});
        O.use_default(P);
      }
    }
  }
  auto program = to_string(Evaluate({O}));
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  X0[X0_0, X0_1, X0_2, X0_3],
  X1[X1_0, X1_1, X1_2]
) -> (
  X2
) {
  X2[x0, 3, x1, x2 : 1, 7, 10, 10] = =(X1[x0, x1, x2]) default X0;
}
)"));
}

Tensor ArgMax(const Tensor& I) {
  auto X0 = I[0], X1 = I[1], X2 = I[2];
  Tensor Max;
  for (const auto x0 : Index()) {
    for (const auto x1 : Index()) {
      for (const auto x2 : Index()) {
        Max({x0, x2}, {X0, X2}) >= I({x0, x1, x2});
      }
    }
  }
  Tensor One(tile::SimpleShape(tile::DataType::FLOAT32, {}));
  Tensor T;
  for (const auto x1 : Index()) {
    T({x1}, {X1}) = One({});
  }
  Tensor IX = index(T, 0);
  Tensor O;
  for (const auto x0 : Index()) {
    for (const auto x1 : Index()) {
      for (const auto x2 : Index()) {
        O({x0, x2}, {X0, X2}) >= cond(I({x0, x1, x2}), Max({x0, x2}), IX({x1}));
      }
    }
  }
  return as_uint(O, 32);
}

TEST(TileCC, ArgMax) {
  Tensor I(tile::SimpleShape(tile::DataType::FLOAT32, {1, 10, 10}));
  auto X = ArgMax(I);
  auto program = to_string(Evaluate({X}));
  IVLOG(1, program);
  EXPECT_THAT(X.shape(), Eq(tile::SimpleShape(tile::DataType::UINT32, {1, 10})));
  EXPECT_THAT(program, Eq(R"(function (
  X0[X0_0, X0_1, X0_2],
  X2[]
) -> (
  X8
) {
  X1[x0, x2 : 1, 10] = >(X0[x0, x1, x2]);
  X3[x0 : 10] = =(X2[]);
  X4 = 0;
  X5 = index(X3, X4);
  X6[x0, x2 : 1, 10] = >(X0[x0, x1, x2] == X1[x0, x2] ? X5[x1]);
  X7 = 32;
  X8 = as_uint(X6, X7);
}
)"));
}

Tensor Winograd(const Tensor& I, const Tensor& K, const Tensor& A, const Tensor& B, const Tensor& G) {
  auto N = I[0], X = I[1], Y = I[2], CI = I[3];
  auto S = K[0], CO = K[3];
  auto BI = A[0], BO = A[1];
  auto XO = (X - S + 1) / 1;
  auto YO = (Y - S + 1) / 1;
  auto XB = (XO + BO - 1) / BO;
  auto YB = (YO + BO - 1) / BO;
  auto XP = 0, YP = 0;
  assert(BI - CI + 1 == BO);
  Tensor U1, U, V1, V, M, O1, O;
  Index n, i, j, k, x, y, ci, co;
  U1({i, j, ci, co}, {BI, S, CI, CO}) += G({i, k}) * K({k, j, ci, co});
  U({i, j, ci, co}, {BI, BI, CI, CO}) += U1({i, k, ci, co}) * G({j, k});
  V1({n, i, j, x, y, ci}, {N, BI, BI, XB, YB, CI}) += B({k, i}) * I({n, BO * x + k - XP, BO * y + j - YP, ci});
  V({n, i, j, x, y, ci}, {N, BI, BI, XB, YB, CI}) += V1({n, i, k, x, y, ci}) * B({k, j});
  M({n, i, j, x, y, co}, {N, BI, BI, XB, YB, CO}) += V({n, i, j, x, y, ci}) * U({i, j, ci, co});
  O1({n, i, j, x, y, co}, {N, BO, BI, XB, YB, CO}) += A({k, i}) * M({n, k, j, x, y, co});
  O({n, BO * x + i, BO * y + j, co}, {N, XO, YO, CO}) += O1({n, i, k, x, y, co}) * A({k, j});
  O.no_defract();
  return O;
}

TEST(TileCC, Wingrad) {
  const size_t N = 1, X = 224, Y = 224, CI = 3, S = 3, CO = 32, BI = 32, BO = BI - CI + 1;
  Tensor I(tile::SimpleShape(tile::DataType::FLOAT32, {N, X, Y, CI}));
  Tensor K(tile::SimpleShape(tile::DataType::FLOAT32, {S, S, CI, CO}));
  Tensor A(tile::SimpleShape(tile::DataType::FLOAT32, {BI, BO}));
  Tensor B(tile::SimpleShape(tile::DataType::FLOAT32, {BI, BI}));
  Tensor G(tile::SimpleShape(tile::DataType::FLOAT32, {BI, S}));
  auto W = Winograd(I, K, A, B, G);
  auto program = to_string(Evaluate({W}));
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  X0[X0_0, X0_1],
  X1[X1_0, X1_1],
  X2[X2_0, X2_1, X2_2, X2_3],
  X5[X5_0, X5_1],
  X6[X6_0, X6_1, X6_2, X6_3]
) -> (
  X11
) {
  X3[x2, x1, x5, x3, x4, x6 : 1, 32, 32, 8, 8, 3] = +(X1[x0, x1] * X2[x2, x0 + 30*x3, 30*x4 + x5, x6]);
  X4[x0, x1, x6, x3, x4, x5 : 1, 32, 32, 8, 8, 3] = +(X3[x0, x1, x2, x3, x4, x5] * X1[x2, x6]);
  X7[x0, x2, x3, x4 : 32, 3, 3, 32] = +(X5[x0, x1] * X6[x1, x2, x3, x4]);
  X8[x0, x4, x2, x3 : 32, 32, 3, 32] = +(X7[x0, x1, x2, x3] * X5[x4, x1]);
  X9[x0, x1, x2, x3, x4, x6 : 1, 32, 32, 8, 8, 32] = +(X4[x0, x1, x2, x3, x4, x5] * X8[x1, x2, x5, x6]);
  X10[x2, x1, x3, x4, x5, x6 : 1, 30, 32, 8, 8, 32] = +(X0[x0, x1] * X9[x2, x0, x3, x4, x5, x6]);
  X11[x0, x1 + 30*x3, 30*x4 + x6, x5 : 1, 222, 222, 32] = +(X10[x0, x1, x2, x3, x4, x5] * X0[x2, x6]) no_defract;
}
)"));
}

}  // namespace
}  // namespace tile_cc
}  // namespace plaidml
}  // namespace vertexai
