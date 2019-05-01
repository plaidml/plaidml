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
  Tensor R;
  for (auto x0 : Index()) {
    for (auto y1 : Index()) {
      for (auto z : Index()) {
        R({x0, y1}, {X.dims(0), Y.dims(1)}) += X({x0, z}) * Y({z, y1});
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
      M({i, 0}, {X.dims(0), 1}) >= X({i, j});
    }
  }
  auto E = exp(X - M);
  Tensor N;
  for (auto i : Index()) {
    for (auto j : Index()) {
      N({i, 0}, {X.dims(0), 1}) += E({i, j});
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
  Tensor R;
  auto N = I.dims(0), H = I.dims(1), W = I.dims(2);
  auto KH = K.dims(0), KW = K.dims(1), CO = K.dims(3);
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
  auto N = I.dims(0), H = I.dims(1), W = I.dims(2), C = I.dims(3);
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
  auto program = to_string(Evaluate("lars_momentum4d", {std::get<0>(R), std::get<1>(R)}).program);
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  _X0[_X0_0, _X0_1, _X0_2, _X0_3],
  _X3[],
  _X6[_X6_0, _X6_1, _X6_2, _X6_3],
  _X11[_X11_0, _X11_1, _X11_2, _X11_3]
) -> (
  _X24,
  _X23
) {
  _X1 = 0.125000;
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
  _X16 = mul(_X9, _X15);
  _X17 = add(_X14, _X16);
  _X18 = div(_X10, _X17);
  _X19 = 0.000488;
  _X20 = mul(_X6, _X19);
  _X21 = add(_X11, _X20);
  _X22 = mul(_X18, _X21);
  _X23 = add(_X2, _X22);
  _X24 = sub(_X6, _X23);
}
)"));
}

TEST(TileCC, RepeatElements) {
  Tensor I(tile::SimpleShape(tile::DataType::FLOAT32, {10, 10, 10}));
  auto N0 = I.dims(0), N1 = I.dims(1), N2 = I.dims(2);
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
  auto B = I.dims(0), N1 = I.dims(1), N2 = I.dims(2);
  Tensor O;
  for (auto b : Index()) {
    for (auto i1 : Index()) {
      for (auto i2 : Index()) {
        O({b, 3, i1, i2}, {B, 7, N1, N2}) = I({b, i1, i2});
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
  auto X0 = I.dims(0), X1 = I.dims(1), X2 = I.dims(2);
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
  auto N = I.dims(0), X = I.dims(1), Y = I.dims(2), CI = I.dims(3);
  auto S = K.dims(0), CO = K.dims(3);
  auto BI = A.dims(0), BO = A.dims(1);
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

}  // namespace
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
