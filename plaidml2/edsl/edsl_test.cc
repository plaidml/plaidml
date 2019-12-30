// Copyright 2019 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/StringRef.h"

#include "base/util/env.h"
#include "base/util/logging.h"
#include "plaidml2/edsl/autodiff.h"
#include "plaidml2/edsl/edsl.h"
#include "plaidml2/exec/exec.h"

using ::testing::ContainerEq;
using ::testing::Eq;

namespace plaidml::edsl {

bool operator==(const Program& lhs, const std::string& rhs) {  //
  return llvm::StringRef(lhs.str()).trim() == llvm::StringRef(rhs).trim();
}

namespace {

Tensor Dot(const Tensor& X, const Tensor& Y) {
  TensorDim I, J, K;
  TensorIndex i("i"), j("j"), k("k");
  X.bind_dims(I, K);
  Y.bind_dims(K, J);
  auto R = TensorOutput(I, J);
  R(i, j) += X(i, k) * Y(k, j);
  return R;
}

Tensor Relu(const Tensor& I) { return select(I < 0.0, Tensor{0.0}, I); }

Tensor Softmax(const Tensor& X) {
  TensorDim I, J;
  TensorIndex i, j;
  X.bind_dims(I, J);
  auto M = TensorOutput(I, 1);
  M(i, 0) >= X(i, j);
  auto E = exp(X - M);
  auto N = TensorOutput(I, 1);
  N(i, 0) += E(i, j);
  return E / N;
}

TEST(CppEdsl, Dot) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {3, 3});
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {3, 3});
  auto C = Dot(A, B);
  Program program("dot", {C});
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
#map0 = (d0, d1, d2) -> (d0, d1)
#map1 = (d0, d1, d2) -> (d0, d2)
#map2 = (d0, d1, d2) -> (d2, d1)


!fp32 = type tensor<!eltwise.fp32>
module {
  func @dot(%arg0: tensor<3x3x!eltwise.fp32>, %arg1: tensor<3x3x!eltwise.fp32>) -> tensor<3x3x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !fp32
    %0 = tile.cion add, mul, %cst, %arg1, %arg0 {idxs = ["i", "j", "k"], sink = #map0, srcs = [#map1, #map2]} : !fp32, tensor<3x3x!eltwise.fp32>, tensor<3x3x!eltwise.fp32> -> tensor<3x3x!eltwise.fp32>
    return %0 : tensor<3x3x!eltwise.fp32>
  }
}
)#"));
#endif

  std::vector<float> input = {
      1.0f, 2.0f, 3.0f,  //
      4.0f, 5.0f, 6.0f,  //
      7.0f, 8.0f, 9.0f,  //
  };

  std::vector<float> expected = {
      30.0f,  36.0f,  42.0f,   //
      66.0f,  81.0f,  96.0f,   //
      102.0f, 126.0f, 150.0f,  //
  };

  auto binder = exec::Binder(program);
  auto executable = binder.compile();
  binder.input(A).copy_from(input.data());
  binder.input(B).copy_from(input.data());
  executable->run();
  {
    auto view = binder.output(C).mmap_current();
    ASSERT_THAT(view.size(), expected.size() * sizeof(expected[0]));
    auto data = reinterpret_cast<float*>(view.data());
    std::vector<float> actual(data, data + expected.size());
    EXPECT_THAT(actual, ContainerEq(expected));
  }
}

TEST(CppEdsl, DoubleDot) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20});
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {20, 30});
  auto C = Placeholder(PLAIDML_DATA_FLOAT32, {30, 40});
  Program program("double_dot", {Dot(Dot(A, B), C)});
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
#map0 = (d0, d1, d2) -> (d0, d1)
#map1 = (d0, d1, d2) -> (d0, d2)
#map2 = (d0, d1, d2) -> (d2, d1)


!fp32 = type tensor<!eltwise.fp32>
module {
  func @double_dot(%arg0: tensor<30x40x!eltwise.fp32>, %arg1: tensor<20x30x!eltwise.fp32>, %arg2: tensor<10x20x!eltwise.fp32>) -> tensor<10x40x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !fp32
    %0 = tile.cion add, mul, %cst, %arg2, %arg1 {idxs = ["i", "j", "k"], sink = #map0, srcs = [#map1, #map2]} : !fp32, tensor<10x20x!eltwise.fp32>, tensor<20x30x!eltwise.fp32> -> tensor<10x30x!eltwise.fp32>
    %1 = tile.cion add, mul, %cst, %0, %arg0 {idxs = ["i", "j", "k"], sink = #map0, srcs = [#map1, #map2]} : !fp32, tensor<10x30x!eltwise.fp32>, tensor<30x40x!eltwise.fp32> -> tensor<10x40x!eltwise.fp32>
    return %1 : tensor<10x40x!eltwise.fp32>
  }
}
)#"));
#endif
  exec::Binder(program).compile()->run();
}  // namespace plaidml::edsl

TEST(CppEdsl, EltwiseAdd) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20});
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20});
  Program program("eltwise_add", {A + B});
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
module {
  func @eltwise_add(%arg0: tensor<10x20x!eltwise.fp32>, %arg1: tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32> {
    %0 = "eltwise.add"(%arg1, %arg0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %0 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
#endif
  exec::Binder(program).compile()->run();
}

TEST(CppEdsl, Relu) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20});
  Program program("relu", {Relu(A)});
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
!fp32 = type tensor<!eltwise.fp32>
module {
  func @relu(%arg0: tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !fp32
    %0 = "eltwise.cmp_lt"(%arg0, %cst) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, !fp32) -> tensor<10x20x!eltwise.bool>
    %1 = "eltwise.select"(%0, %cst, %arg0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.bool>, !fp32, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %1 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
#endif
  exec::Binder(program).compile()->run();
}

TEST(CppEdsl, MnistMlp) {
  // model.add(Dense(512, activation='relu', input_shape=(784,)))
  auto input = Placeholder(PLAIDML_DATA_FLOAT32, {1, 784});
  auto kernel1 = Placeholder(PLAIDML_DATA_FLOAT32, {784, 512});
  auto bias1 = Placeholder(PLAIDML_DATA_FLOAT32, {512});
  auto dense1 = Relu(Dot(input, kernel1) + bias1);
  // model.add(Dense(512, activation='relu'))
  auto kernel2 = Placeholder(PLAIDML_DATA_FLOAT32, {512, 512});
  auto bias2 = Placeholder(PLAIDML_DATA_FLOAT32, {512});
  auto dense2 = Relu(Dot(dense1, kernel2) + bias2);
  // model.add(Dense(10, activation='softmax'))
  auto kernel3 = Placeholder(PLAIDML_DATA_FLOAT32, {512, 10});
  auto bias3 = Placeholder(PLAIDML_DATA_FLOAT32, {10});
  auto dense3 = Softmax(Dot(dense2, kernel3) + bias3);
  Program program("mnist_mlp", {dense3});
#ifdef PLAIDML_AST
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
  _X2[i, j : 1, 512] = +(_X0[i, k] * _X1[k, j]);
  _X4 = add(_X2, _X3);
  _X5 = 0.000000;
  _X6 = cmp_lt(_X4, _X5);
  _X7 = 0.000000;
  _X8 = cond(_X6, _X7, _X4);
  _X10[i, j : 1, 512] = +(_X8[i, k] * _X9[k, j]);
  _X12 = add(_X10, _X11);
  _X13 = 0.000000;
  _X14 = cmp_lt(_X12, _X13);
  _X15 = 0.000000;
  _X16 = cond(_X14, _X15, _X12);
  _X18[i, j : 1, 10] = +(_X16[i, k] * _X17[k, j]);
  _X20 = add(_X18, _X19);
  _X21[x0, 0 : 1, 1] = >(_X20[x0, x1]);
  _X22 = sub(_X20, _X21);
  _X23 = exp(_X22);
  _X24[x0, 0 : 1, 1] = +(_X23[x0, x1]);
  _X25 = div(_X23, _X24);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
#map0 = (d0, d1, d2) -> (d0, d1)
#map1 = (d0, d1, d2) -> (d0, d2)
#map2 = (d0, d1, d2) -> (d2, d1)
#map3 = (d0, d1) -> (d0, 0)
#map4 = (d0, d1) -> (d0, d1)


!fp32 = type tensor<!eltwise.fp32>
module {
  func @mnist_mlp(%arg0: tensor<10x!eltwise.fp32>, %arg1: tensor<512x10x!eltwise.fp32>, %arg2: tensor<512x!eltwise.fp32>, %arg3: tensor<512x512x!eltwise.fp32>, %arg4: tensor<512x!eltwise.fp32>, %arg5: tensor<784x512x!eltwise.fp32>, %arg6: tensor<1x784x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !fp32
    %0 = tile.cion add, mul, %cst, %arg6, %arg5 {idxs = ["i", "j", "k"], sink = #map0, srcs = [#map1, #map2]} : !fp32, tensor<1x784x!eltwise.fp32>, tensor<784x512x!eltwise.fp32> -> tensor<1x512x!eltwise.fp32>
    %1 = "eltwise.add"(%0, %arg4) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.fp32>, tensor<512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32>
    %2 = "eltwise.cmp_lt"(%1, %cst) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.fp32>, !fp32) -> tensor<1x512x!eltwise.bool>
    %3 = "eltwise.select"(%2, %cst, %1) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.bool>, !fp32, tensor<1x512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32>
    %4 = tile.cion add, mul, %cst, %3, %arg3 {idxs = ["i", "j", "k"], sink = #map0, srcs = [#map1, #map2]} : !fp32, tensor<1x512x!eltwise.fp32>, tensor<512x512x!eltwise.fp32> -> tensor<1x512x!eltwise.fp32>
    %5 = "eltwise.add"(%4, %arg2) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.fp32>, tensor<512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32>
    %6 = "eltwise.cmp_lt"(%5, %cst) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.fp32>, !fp32) -> tensor<1x512x!eltwise.bool>
    %7 = "eltwise.select"(%6, %cst, %5) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.bool>, !fp32, tensor<1x512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32>
    %8 = tile.cion add, mul, %cst, %7, %arg1 {idxs = ["i", "j", "k"], sink = #map0, srcs = [#map1, #map2]} : !fp32, tensor<1x512x!eltwise.fp32>, tensor<512x10x!eltwise.fp32> -> tensor<1x10x!eltwise.fp32>
    %9 = "eltwise.add"(%8, %arg0) {type = !eltwise.fp32} : (tensor<1x10x!eltwise.fp32>, tensor<10x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32>
    %10 = tile.cion max, none, %cst, %9 {sink = #map3, srcs = [#map4]} : !fp32, tensor<1x10x!eltwise.fp32> -> tensor<1x1x!eltwise.fp32>
    %11 = "eltwise.sub"(%9, %10) {type = !eltwise.fp32} : (tensor<1x10x!eltwise.fp32>, tensor<1x1x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32>
    %12 = "eltwise.exp"(%11) {type = !eltwise.fp32} : (tensor<1x10x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32>
    %13 = tile.cion add, none, %cst, %12 {sink = #map3, srcs = [#map4]} : !fp32, tensor<1x10x!eltwise.fp32> -> tensor<1x1x!eltwise.fp32>
    %14 = "eltwise.div"(%12, %13) {type = !eltwise.fp32} : (tensor<1x10x!eltwise.fp32>, tensor<1x1x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32>
    return %14 : tensor<1x10x!eltwise.fp32>
  }
}
)#"));
#endif
  exec::Binder(program).compile()->run();
}

Tensor Convolution2(const Tensor& I, const Tensor& K) {
  TensorDim CI, CO, K0, K1, N, X0, X1;
  TensorIndex n, x0, x1, co, ci, k0, k1;
  I.bind_dims(N, X0, X1, CI);
  K.bind_dims(K0, K1, CI, CO);
  auto R = TensorOutput(N, X0 - (K0 - 1), X1 - (K1 - 1), CO);
  R(n, x0, x1, co) += I(n, x0 + k0 - (K0 / 2), x1 + k1 - (K1 / 2), ci) * K(k0, k1, ci, co);
  return R;
}

TEST(CppEdsl, Convolution) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 1});
  auto K = Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 1, 32});
  Program program("convolution", {Convolution2(I, K)});
#ifdef PLAIDML_MLIR
  exec::Binder(program).compile()->run();
#endif
}

Tensor MaxPooling2(const Tensor& I) {
  TensorDim N, X0, X1, C;
  TensorIndex n, x0, x1, i, j, c;
  I.bind_dims(N, X0, X1, C);
  auto R = TensorOutput(N, (X0 + 1) / 2, (X1 + 1) / 2, C);
  R(n, x0, x1, c) >= I(n, 2 * x0 + i, 2 * x1 + j, c);
  R.add_constraints({i < 2, j < 2});
  return R;
}

Tensor Flatten(const Tensor& X) {
  std::vector<TensorDim> X_dims(X.shape().ndims());
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

TEST(CppEdsl, MnistCnn) {
  // model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
  auto input = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 1});
  auto kernel1 = Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 1, 32});
  auto bias1 = Placeholder(PLAIDML_DATA_FLOAT32, {32});
  auto conv1 = Relu(Convolution2(input, kernel1) + bias1);
  // model.add(Conv2D(64, (3, 3), activation='relu'))
  auto kernel2 = Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 32, 64});
  auto bias2 = Placeholder(PLAIDML_DATA_FLOAT32, {64});
  auto conv2 = Relu(Convolution2(conv1, kernel2) + bias2);
  // model.add(MaxPooling2D(pool_size=(2, 2)))
  auto pool1 = MaxPooling2(conv2);
  // model.add(Flatten())
  auto flat = Flatten(pool1);
  EXPECT_THAT(flat.shape(), Eq(LogicalShape(PLAIDML_DATA_FLOAT32, {1, 12100})));
  // model.add(Dense(128, activation='relu'))
  auto kernel3 = Placeholder(PLAIDML_DATA_FLOAT32, {12100, 128});
  auto bias3 = Placeholder(PLAIDML_DATA_FLOAT32, {128});
  auto dense1 = Relu(Dot(flat, kernel3) + bias3);
  const std::int64_t kNumClasses = 100;
  // model.add(Dense(num_classes, activation='softmax'))
  auto kernel4 = Placeholder(PLAIDML_DATA_FLOAT32, {128, kNumClasses});
  auto bias4 = Placeholder(PLAIDML_DATA_FLOAT32, {kNumClasses});
  auto dense2 = Softmax(Dot(dense1, kernel4) + bias4);
  Program program("mnist_cnn", {dense2});
#ifdef PLAIDML_AST
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
  _X22[i, j : 1, 128] = +(_X20[i, k] * _X21[k, j]);
  _X24 = add(_X22, _X23);
  _X25 = 0.000000;
  _X26 = cmp_lt(_X24, _X25);
  _X27 = 0.000000;
  _X28 = cond(_X26, _X27, _X24);
  _X30[i, j : 1, 100] = +(_X28[i, k] * _X29[k, j]);
  _X32 = add(_X30, _X31);
  _X33[x0, 0 : 1, 1] = >(_X32[x0, x1]);
  _X34 = sub(_X32, _X33);
  _X35 = exp(_X34);
  _X36[x0, 0 : 1, 1] = +(_X35[x0, x1]);
  _X37 = div(_X35, _X36);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
#map0 = (d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)
#map1 = (d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 - 1, d2 + d5 - 1, d6)
#map2 = (d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)
#map3 = (d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)
#map4 = (d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d3)
#map5 = (d0, d1, d2) -> (d0, d1)
#map6 = (d0, d1, d2) -> (d0, d2)
#map7 = (d0, d1, d2) -> (d2, d1)
#map8 = (d0, d1) -> (d0, 0)
#map9 = (d0, d1) -> (d0, d1)

#set0 = (d0, d1, d2, d3, d4, d5) : (d4 >= 0, -d4 + 1 >= 0, d5 >= 0, -d5 + 1 >= 0)

!fp32 = type tensor<!eltwise.fp32>
module {
  func @mnist_cnn(%arg0: tensor<100x!eltwise.fp32>, %arg1: tensor<128x100x!eltwise.fp32>, %arg2: tensor<128x!eltwise.fp32>, %arg3: tensor<12100x128x!eltwise.fp32>, %arg4: tensor<64x!eltwise.fp32>, %arg5: tensor<3x3x32x64x!eltwise.fp32>, %arg6: tensor<32x!eltwise.fp32>, %arg7: tensor<3x3x1x32x!eltwise.fp32>, %arg8: tensor<1x224x224x1x!eltwise.fp32>) -> tensor<1x100x!eltwise.fp32> {
    %c12100 = tile.affine_const 12100
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !fp32
    %c1 = tile.affine_const 1
    %0 = tile.cion add, mul, %cst, %arg8, %arg7 {sink = #map0, srcs = [#map1, #map2]} : !fp32, tensor<1x224x224x1x!eltwise.fp32>, tensor<3x3x1x32x!eltwise.fp32> -> tensor<1x222x222x32x!eltwise.fp32>
    %1 = "eltwise.add"(%0, %arg6) {type = !eltwise.fp32} : (tensor<1x222x222x32x!eltwise.fp32>, tensor<32x!eltwise.fp32>) -> tensor<1x222x222x32x!eltwise.fp32>
    %2 = "eltwise.cmp_lt"(%1, %cst) {type = !eltwise.fp32} : (tensor<1x222x222x32x!eltwise.fp32>, !fp32) -> tensor<1x222x222x32x!eltwise.bool>
    %3 = "eltwise.select"(%2, %cst, %1) {type = !eltwise.fp32} : (tensor<1x222x222x32x!eltwise.bool>, !fp32, tensor<1x222x222x32x!eltwise.fp32>) -> tensor<1x222x222x32x!eltwise.fp32>
    %4 = tile.cion add, mul, %cst, %3, %arg5 {sink = #map0, srcs = [#map1, #map2]} : !fp32, tensor<1x222x222x32x!eltwise.fp32>, tensor<3x3x32x64x!eltwise.fp32> -> tensor<1x220x220x64x!eltwise.fp32>
    %5 = "eltwise.add"(%4, %arg4) {type = !eltwise.fp32} : (tensor<1x220x220x64x!eltwise.fp32>, tensor<64x!eltwise.fp32>) -> tensor<1x220x220x64x!eltwise.fp32>
    %6 = "eltwise.cmp_lt"(%5, %cst) {type = !eltwise.fp32} : (tensor<1x220x220x64x!eltwise.fp32>, !fp32) -> tensor<1x220x220x64x!eltwise.bool>
    %7 = "eltwise.select"(%6, %cst, %5) {type = !eltwise.fp32} : (tensor<1x220x220x64x!eltwise.bool>, !fp32, tensor<1x220x220x64x!eltwise.fp32>) -> tensor<1x220x220x64x!eltwise.fp32>
    %8 = tile.cion max, none, %cst, %7 {cons = #set0, sink = #map3, srcs = [#map4]} : !fp32, tensor<1x220x220x64x!eltwise.fp32> -> tensor<1x110x110x64x!eltwise.fp32>
    %9 = "tile.reshape"(%8, %c1, %c12100) : (tensor<1x110x110x64x!eltwise.fp32>, index, index) -> tensor<1x12100x!eltwise.fp32>
    %10 = tile.cion add, mul, %cst, %9, %arg3 {idxs = ["i", "j", "k"], sink = #map5, srcs = [#map6, #map7]} : !fp32, tensor<1x12100x!eltwise.fp32>, tensor<12100x128x!eltwise.fp32> -> tensor<1x128x!eltwise.fp32>
    %11 = "eltwise.add"(%10, %arg2) {type = !eltwise.fp32} : (tensor<1x128x!eltwise.fp32>, tensor<128x!eltwise.fp32>) -> tensor<1x128x!eltwise.fp32>
    %12 = "eltwise.cmp_lt"(%11, %cst) {type = !eltwise.fp32} : (tensor<1x128x!eltwise.fp32>, !fp32) -> tensor<1x128x!eltwise.bool>
    %13 = "eltwise.select"(%12, %cst, %11) {type = !eltwise.fp32} : (tensor<1x128x!eltwise.bool>, !fp32, tensor<1x128x!eltwise.fp32>) -> tensor<1x128x!eltwise.fp32>
    %14 = tile.cion add, mul, %cst, %13, %arg1 {idxs = ["i", "j", "k"], sink = #map5, srcs = [#map6, #map7]} : !fp32, tensor<1x128x!eltwise.fp32>, tensor<128x100x!eltwise.fp32> -> tensor<1x100x!eltwise.fp32>
    %15 = "eltwise.add"(%14, %arg0) {type = !eltwise.fp32} : (tensor<1x100x!eltwise.fp32>, tensor<100x!eltwise.fp32>) -> tensor<1x100x!eltwise.fp32>
    %16 = tile.cion max, none, %cst, %15 {sink = #map8, srcs = [#map9]} : !fp32, tensor<1x100x!eltwise.fp32> -> tensor<1x1x!eltwise.fp32>
    %17 = "eltwise.sub"(%15, %16) {type = !eltwise.fp32} : (tensor<1x100x!eltwise.fp32>, tensor<1x1x!eltwise.fp32>) -> tensor<1x100x!eltwise.fp32>
    %18 = "eltwise.exp"(%17) {type = !eltwise.fp32} : (tensor<1x100x!eltwise.fp32>) -> tensor<1x100x!eltwise.fp32>
    %19 = tile.cion add, none, %cst, %18 {sink = #map8, srcs = [#map9]} : !fp32, tensor<1x100x!eltwise.fp32> -> tensor<1x1x!eltwise.fp32>
    %20 = "eltwise.div"(%18, %19) {type = !eltwise.fp32} : (tensor<1x100x!eltwise.fp32>, tensor<1x1x!eltwise.fp32>) -> tensor<1x100x!eltwise.fp32>
    return %20 : tensor<1x100x!eltwise.fp32>
  }
}
)#"));
  // exec::Binder(program).compile()->run();
#endif  // PLAIDML_MLIR
}

Tensor Normalize(const Tensor& X) {
  auto XSqr = X * X;
  auto X_MS = TensorOutput();
  std::vector<TensorIndex> idxs(X.shape().ndims());
  X_MS() += XSqr(idxs);
  return sqrt(X_MS);
}

std::tuple<Tensor, Tensor> LarsMomentum(  //
    const Tensor& X,                      //
    const Tensor& Grad,                   //
    const Tensor& Veloc,                  //
    const Tensor& LR,                     //
    double lars_coeff,                    //
    double lars_weight_decay,             //
    double momentum) {
  auto XNorm = Normalize(X);
  auto GradNorm = Normalize(Grad);
  auto LocLR = LR * lars_coeff * XNorm / (GradNorm + lars_weight_decay * XNorm);
  auto NewVeloc = momentum * Veloc + LocLR * (Grad + lars_weight_decay * X);
  return std::make_tuple(X - NewVeloc, NewVeloc);
}

TEST(CppEdsl, LarsMomentum4d) {
  auto X_shape = LogicalShape(PLAIDML_DATA_FLOAT32, {4, 7, 3, 9});
  auto LR_shape = LogicalShape(PLAIDML_DATA_FLOAT32, {});
  auto X = Placeholder(X_shape);
  auto Grad = Placeholder(X_shape);
  auto Veloc = Placeholder(X_shape);
  auto LR = Placeholder(LR_shape);
  auto R = LarsMomentum(X, Grad, Veloc, LR, 1. / 1024., 1. / 2048., 1. / 8.);
  Program program("lars_momentum4d", {std::get<0>(R), std::get<1>(R)});
#ifdef PLAIDML_AST
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
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
#map0 = () -> ()
#map1 = (d0, d1, d2, d3) -> (d0, d1, d2, d3)


!fp32 = type tensor<!eltwise.fp32>
module {
  func @lars_momentum4d(%arg0: tensor<4x7x3x9x!eltwise.fp32>, %arg1: tensor<4x7x3x9x!eltwise.fp32>, %arg2: !fp32, %arg3: tensor<4x7x3x9x!eltwise.fp32>) -> (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !fp32
    %cst_0 = "eltwise.sconst"() {value = 4.8828125E-4 : f64} : () -> !fp32
    %cst_1 = "eltwise.sconst"() {value = 9.765625E-4 : f64} : () -> !fp32
    %cst_2 = "eltwise.sconst"() {value = 1.250000e-01 : f64} : () -> !fp32
    %0 = "eltwise.mul"(%arg0, %cst_0) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, !fp32) -> tensor<4x7x3x9x!eltwise.fp32>
    %1 = "eltwise.add"(%arg1, %0) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %2 = "eltwise.mul"(%arg0, %arg0) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %3 = tile.cion add, none, %cst, %2 {sink = #map0, srcs = [#map1]} : !fp32, tensor<4x7x3x9x!eltwise.fp32> -> !fp32
    %4 = "eltwise.sqrt"(%3) {type = !eltwise.fp32} : (!fp32) -> !fp32
    %5 = "eltwise.mul"(%4, %cst_0) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %6 = "eltwise.mul"(%arg1, %arg1) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %7 = tile.cion add, none, %cst, %6 {sink = #map0, srcs = [#map1]} : !fp32, tensor<4x7x3x9x!eltwise.fp32> -> !fp32
    %8 = "eltwise.sqrt"(%7) {type = !eltwise.fp32} : (!fp32) -> !fp32
    %9 = "eltwise.add"(%8, %5) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %10 = "eltwise.mul"(%arg2, %cst_1) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %11 = "eltwise.mul"(%10, %4) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %12 = "eltwise.div"(%11, %9) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %13 = "eltwise.mul"(%12, %1) {type = !eltwise.fp32} : (!fp32, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %14 = "eltwise.mul"(%arg3, %cst_2) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, !fp32) -> tensor<4x7x3x9x!eltwise.fp32>
    %15 = "eltwise.add"(%14, %13) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %16 = "eltwise.sub"(%arg0, %15) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    return %16, %15 : tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>
  }
}
)#"));
#endif
  exec::Binder(program).compile()->run();
}

TEST(CppEdsl, RepeatElements) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 10, 10});
  TensorDim N0, N1, N2;
  TensorIndex n0, n1, n2, k;
  I.bind_dims(N0, N1, N2);
  auto O = TensorOutput(N0, 3 * N1, N2);
  O(n0, 3 * n1 + k, n2) = I(n0, n1, n2);
  O.add_constraint(k < 3);
  O.no_reduce();
  Program program("repeat_elts", {O});
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  _X0[_X0_0, _X0_1, _X0_2]
) -> (
  _X1
) {
  _X1[x0, 3*x1 + x3, x2 : 10, 30, 10] = =(_X0[x0, x1, x2]), x3 < 3 no_defract;
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
#map0 = (d0, d1, d2, d3) -> (d0, d1 * 3 + d2, d3)
#map1 = (d0, d1, d2, d3) -> (d0, d1, d3)

#set0 = (d0, d1, d2, d3) : (d2 >= 0, -d2 + 2 >= 0)

!fp32 = type tensor<!eltwise.fp32>
module {
  func @repeat_elts(%arg0: tensor<10x10x10x!eltwise.fp32>) -> tensor<10x30x10x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !fp32
    %0 = tile.cion assign, none, %cst, %arg0 {cons = #set0, no_reduce, sink = #map0, srcs = [#map1]} : !fp32, tensor<10x10x10x!eltwise.fp32> -> tensor<10x30x10x!eltwise.fp32>
    return %0 : tensor<10x30x10x!eltwise.fp32>
  }
}
)#"));
#endif
  exec::Binder(program).compile()->run();
}

TEST(CppEdsl, UseDefault) {
  auto P = Placeholder(PLAIDML_DATA_FLOAT32, {1, 7, 10, 10});
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 10, 10});
  TensorDim B, N1, N2;
  TensorIndex b, i1, i2;
  I.bind_dims(B, N1, N2);
  auto O = TensorOutput(B, 7, N1, N2);
  O(b, 3, i1, i2) = I(b, i1, i2);
  O.use_default(P);
  Program program("use_default", {O});
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  _X0[_X0_0, _X0_1, _X0_2, _X0_3],
  _X1[_X1_0, _X1_1, _X1_2]
) -> (
  _X2
) {
  _X2[x0, 3, x1, x2 : 1, 7, 10, 10] = =(_X1[x0, x1, x2]) default _X0;
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
#map0 = (d0, d1, d2) -> (d0, 3, d1, d2)
#map1 = (d0, d1, d2) -> (d0, d1, d2)


module {
  func @use_default(%arg0: tensor<1x10x10x!eltwise.fp32>, %arg1: tensor<1x7x10x10x!eltwise.fp32>) -> tensor<1x7x10x10x!eltwise.fp32> {
    %0 = tile.cion assign, none, %arg1, %arg0 {sink = #map0, srcs = [#map1]} : tensor<1x7x10x10x!eltwise.fp32>, tensor<1x10x10x!eltwise.fp32> -> tensor<1x7x10x10x!eltwise.fp32>
    return %0 : tensor<1x7x10x10x!eltwise.fp32>
  }
}
)#"));
#endif
  exec::Binder(program).compile()->run();
}

Tensor ArgMax(const Tensor& I) {
  TensorDim X0, X1, X2;
  TensorIndex x0, x1, x2;
  I.bind_dims(X0, X1, X2);
  auto Max = TensorOutput(X0, X2);
  Max(x0, x2) >= I(x0, x1, x2);
  Tensor One{1};
  auto T = TensorOutput(X1);
  T(x1) = One();
  Tensor IX = index(T, 0);
  auto O = TensorOutput(X0, X2);
  O(x0, x2) >= cond(I(x0, x1, x2), Max(x0, x2), IX(x1));
  return as_uint(O, 32);
}

TEST(CppEdsl, ArgMax) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 10, 10});
  auto X = ArgMax(I);
  Program program("arg_max", {X});
  EXPECT_THAT(X.shape(), Eq(LogicalShape(PLAIDML_DATA_UINT32, {1, 10})));
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  _X0[_X0_0, _X0_1, _X0_2]
) -> (
  _X8
) {
  _X1[x0, x2 : 1, 10] = >(_X0[x0, x1, x2]);
  _X2 = 1;
  _X3[x0 : 10] = =(_X2[]);
  _X4 = 0;
  _X5 = index(_X3, _X4);
  _X6[x0, x2 : 1, 10] = >(_X0[x0, x1, x2] == _X1[x0, x2] ? _X5[x1]);
  _X7 = 32;
  _X8 = as_uint(_X6, _X7);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
#map0 = (d0) -> (d0)
#map1 = () -> ()
#map2 = (d0, d1, d2) -> (d0, d1)
#map3 = (d0, d1, d2) -> (d0, d2, d1)
#map4 = (d0, d1, d2) -> (d2)


!i32 = type tensor<!eltwise.i32>
!fp32 = type tensor<!eltwise.fp32>
module {
  func @arg_max(%arg0: tensor<1x10x10x!eltwise.fp32>) -> tensor<1x10x!eltwise.u32> {
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> !i32
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !fp32
    %0 = tile.cion assign, none, %cst, %c1 {sink = #map0, srcs = [#map1]} : !fp32, !i32 -> tensor<10x!eltwise.i32>
    %1 = "tile.index"(%0) {dim = 0 : i64} : (tensor<10x!eltwise.i32>) -> tensor<10x!eltwise.i32>
    %2 = tile.cion max, none, %cst, %arg0 {sink = #map2, srcs = [#map3]} : !fp32, tensor<1x10x10x!eltwise.fp32> -> tensor<1x10x!eltwise.fp32>
    %3 = tile.cion max, cond, %cst, %arg0, %2, %1 {sink = #map2, srcs = [#map3, #map2, #map4]} : !fp32, tensor<1x10x10x!eltwise.fp32>, tensor<1x10x!eltwise.fp32>, tensor<10x!eltwise.i32> -> tensor<1x10x!eltwise.i32>
    %4 = "eltwise.cast"(%3) : (tensor<1x10x!eltwise.i32>) -> tensor<1x10x!eltwise.u32>
    return %4 : tensor<1x10x!eltwise.u32>
  }
}
)#"));
#endif
  // TODO: cpu backend is missing cast ops (as_uint)
  exec::Binder(program).compile()->run();
}

Tensor Winograd(const Tensor& I, const Tensor& K, const Tensor& A, const Tensor& B, const Tensor& G) {
  TensorDim N, S, X, Y, CI, CO, BI, BO;
  I.bind_dims(N, X, Y, CI);
  K.bind_dims(S, S, CI, CO);
  A.bind_dims(BI, BO);
  B.bind_dims(BI, BI);
  G.bind_dims(BI, S);
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
  O.no_reduce();
  return O;
}

TEST(CppEdsl, Winograd) {
  const std::int64_t N = 1, X = 224, Y = 224, CI = 3, S = 3, CO = 32, BI = 32, BO = BI - CI + 1;
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {N, X, Y, CI});
  auto K = Placeholder(PLAIDML_DATA_FLOAT32, {S, S, CI, CO});
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {BI, BO});
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {BI, BI});
  auto G = Placeholder(PLAIDML_DATA_FLOAT32, {BI, S});
  auto W = Winograd(I, K, A, B, G);
  Program program("winograd", {W});
#ifdef PLAIDML_AST
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
#endif
  exec::Binder(program).compile()->run();
}

TEST(CppEdsl, UniqueNames) {
  LogicalShape shape(PLAIDML_DATA_FLOAT32, {});
  auto A = Placeholder(shape, "A");
  auto B = Placeholder(shape, "B");
  auto C0 = Placeholder(shape, "C");
  auto C1 = Placeholder(shape, "C");
  Program program("unique_names", {A + B + C0 + C1});
#ifdef PLAIDML_AST
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
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
!fp32 = type tensor<!eltwise.fp32>
module {
  func @unique_names(%arg0: !fp32 {tile.name = "C"}, %arg1: !fp32 {tile.name = "C_0"}, %arg2: !fp32 {tile.name = "B"}, %arg3: !fp32 {tile.name = "A"}) -> !fp32 {
    %0 = "eltwise.add"(%arg3, %arg2) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %1 = "eltwise.add"(%0, %arg1) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %2 = "eltwise.add"(%1, %arg0) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    return %2 : !fp32
  }
}
)#"));
#endif
  exec::Binder(program).compile()->run();
}

TEST(CppEdsl, GlobalMin) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 10, 10}, "I");
  TensorIndex i, j, k;
  auto O_Neg = TensorOutput();
  auto Neg = -I;
  O_Neg() >= Neg(i, j, k);
  auto O = -O_Neg;
  Program program("global_min", {O});
#ifdef PLAIDML_AST
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
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
#map0 = () -> ()
#map1 = (d0, d1, d2) -> (d0, d1, d2)


!fp32 = type tensor<!eltwise.fp32>
module {
  func @global_min(%arg0: tensor<10x10x10x!eltwise.fp32> {tile.name = "I"}) -> !fp32 {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !fp32
    %0 = "eltwise.neg"(%arg0) {type = !eltwise.fp32} : (tensor<10x10x10x!eltwise.fp32>) -> tensor<10x10x10x!eltwise.fp32>
    %1 = tile.cion max, none, %cst, %0 {sink = #map0, srcs = [#map1]} : !fp32, tensor<10x10x10x!eltwise.fp32> -> !fp32
    %2 = "eltwise.neg"(%1) {type = !eltwise.fp32} : (!fp32) -> !fp32
    return %2 : !fp32
  }
}
)#"));
#endif
  exec::Binder(program).compile()->run();
}

TEST(CppEdsl, CumSum) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10}, "I");
  TensorDim N;
  TensorIndex i, k;
  I.bind_dims(N);
  auto O = TensorOutput(N);
  O(i) += I(k);
  O.add_constraint(i - k < N);
  Program program("cumsum", {O});
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0]
) -> (
  _X0
) {
  _X0[x1 : 10] = +(I[x0]), -x0 + x1 < 10;
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
#map0 = (d0, d1) -> (d0)
#map1 = (d0, d1) -> (d1)

#set0 = (d0, d1) : (d0 - d1 >= 0, -d0 + d1 + 9 >= 0)

!fp32 = type tensor<!eltwise.fp32>
module {
  func @cumsum(%arg0: tensor<10x!eltwise.fp32> {tile.name = "I"}) -> tensor<10x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !fp32
    %0 = tile.cion add, none, %cst, %arg0 {cons = #set0, sink = #map0, srcs = [#map1]} : !fp32, tensor<10x!eltwise.fp32> -> tensor<10x!eltwise.fp32>
    return %0 : tensor<10x!eltwise.fp32>
  }
}
)#"));
#endif
  exec::Binder(program).compile()->run();
}

Tensor ComplexConv2d(              //
    const Tensor& I,               //
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
  // Specify the output size
  auto O = TensorOutput(N, Y[0], Y[1], G, GCO);
  // Compute the convolution
  O(n, x[0], x[1], g, gco) +=
      I(n, s[0] * x[0] + d[0] * k[0] - P[0], s[1] * x[1] + d[1] * k[1] - P[1], g, gci) * K(k[0], k[1], g, gci, gco);
  return O;
}

TEST(CppEdsl, ComplexConv2d) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3, 3});
  auto K = Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 3, 3, 32});
  auto O = ComplexConv2d(I, K, {2, 2}, {3, 3});
  Program program("complex_conv_2d", {O});
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  _X0[_X0_0, _X0_1, _X0_2, _X0_3, _X0_4],
  _X1[_X1_0, _X1_1, _X1_2, _X1_3, _X1_4]
) -> (
  _X2
) {
  _X2[x0, x1, x3, x5, x7 : 1, 112, 112, 3, 32] = +(_X0[x0, -2 + 2*x1 + 3*x2, -2 + 2*x3 + 3*x4, x5, x6] * _X1[x2, x4, x5, x6, x7]);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
#map0 = (d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)
#map1 = (d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5 * 3 - 2, d2 * 2 + d6 * 3 - 2, d3, d7)
#map2 = (d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d3, d7, d4)


!fp32 = type tensor<!eltwise.fp32>
module {
  func @complex_conv_2d(%arg0: tensor<3x3x3x3x32x!eltwise.fp32>, %arg1: tensor<1x224x224x3x3x!eltwise.fp32>) -> tensor<1x112x112x3x32x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !fp32
    %0 = tile.cion add, mul, %cst, %arg1, %arg0 {sink = #map0, srcs = [#map1, #map2]} : !fp32, tensor<1x224x224x3x3x!eltwise.fp32>, tensor<3x3x3x3x32x!eltwise.fp32> -> tensor<1x112x112x3x32x!eltwise.fp32>
    return %0 : tensor<1x112x112x3x32x!eltwise.fp32>
  }
}
)#"));
#endif
  exec::Binder(program).compile()->run();
}

TEST(CppEdsl, Reciprocal) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10}, "A");
  Program program("reciprocal", {1 / A});
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0]
) -> (
  _X1
) {
  _X0 = 1;
  _X1 = div(_X0, A);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
!i32 = type tensor<!eltwise.i32>
module {
  func @reciprocal(%arg0: tensor<10x!eltwise.fp32> {tile.name = "A"}) -> tensor<10x!eltwise.fp32> {
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> !i32
    %0 = "eltwise.div"(%c1, %arg0) {type = !eltwise.fp32} : (!i32, tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32>
    return %0 : tensor<10x!eltwise.fp32>
  }
}
)#"));
#endif
  exec::Binder(program).compile()->run();
}

#ifdef PLAIDML_AST
TEST(CppEdsl, GradientDot) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {100, 100}, "A");
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {100, 100}, "B");
  auto O = Dot(A, B);
  auto grads = Gradient({A, B}, O);
  Program program("gradient_dot", {grads});
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1],
  B[B_0, B_1]
) -> (
  _X3,
  _X2
) {
  _X0 = 1.000000;
  _X1[x0, x1 : 100, 100] = +(_X0[]);
  _X2[k, j : 100, 100] = +(A[i, k] * _X1[i, j]);
  _X3[i, k : 100, 100] = +(_X1[i, j] * B[k, j]);
}
)"));
  exec::Binder(program).compile()->run();
}
#endif

#ifdef PLAIDML_AST
Tensor Max2Da0(const Tensor& A) {
  TensorDim M, N;
  A.bind_dims(M, N);
  TensorIndex m("m"), n("n");
  auto O = NamedTensorOutput("O", N);
  O(n) >= A(m, n);
  // O(n) += A(m, n);
  return O;
}

TEST(CppEdsl, GradientMultiDot) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {100, 100}, "A");
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {100, 100}, "B");
  auto C = Dot(A, B);
  auto D = Dot(A, C);
  auto O = Max2Da0(D);
  auto grads = Gradient({A, B}, O);
  Program program("gradient_dot", {grads});
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1],
  B[B_0, B_1]
) -> (
  _X9,
  _X6
) {
  _X0[i, j : 100, 100] = +(A[i, k] * B[k, j]);
  _X1[i, j : 100, 100] = +(A[i, k] * _X0[k, j]);
  O[n : 100] = >(_X1[m, n]);
  _X2 = 1.000000;
  _X3[x0 : 100] = +(_X2[]);
  _X4[m, n : 100, 100] = +(_X1[m, n] == O[n] ? _X3[n]);
  _X5[k, j : 100, 100] = +(A[i, k] * _X4[i, j]);
  _X6[k, j : 100, 100] = +(A[i, k] * _X5[i, j]);
  _X7[i, k : 100, 100] = +(_X4[i, j] * _X0[k, j]);
  _X8[i, k : 100, 100] = +(_X5[i, j] * B[k, j]);
  _X9 = add(_X7, _X8);
}
)"));
  exec::Binder(program).compile()->run();
}
#endif

#ifdef PLAIDML_AST
TEST(CppEdsl, GradientDotSqrt) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {100, 100}, "A");
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {100, 100}, "B");
  auto C = Dot(A, B);
  auto O = sqrt(C);
  auto grads = Gradient({A, B}, O);
  Program program("gradient_dot", {grads});
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1],
  B[B_0, B_1]
) -> (
  _X8,
  _X7
) {
  _X0 = 1.000000;
  _X1[x0, x1 : 100, 100] = +(_X0[]);
  _X2 = 2;
  _X3[i, j : 100, 100] = +(A[i, k] * B[k, j]);
  _X4 = sqrt(_X3);
  _X5 = mul(_X2, _X4);
  _X6 = div(_X1, _X5);
  _X7[k, j : 100, 100] = +(A[i, k] * _X6[i, j]);
  _X8[i, k : 100, 100] = +(_X6[i, j] * B[k, j]);
}
)"));
  exec::Binder(program).compile()->run();
}
#endif

TEST(CppEdsl, DefractLong) {
  std::vector<int64_t> input_shape{1, 3, 3, 1};
  std::vector<int64_t> output_shape{1, 5, 5, 1};
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, input_shape, "I");
  auto K = Placeholder(PLAIDML_DATA_FLOAT32, input_shape, "K");
  auto O = TensorOutput(output_shape);
  TensorIndex n, x0, x1, k0, k1, co, ci;
  O(n, x0, x1, co) += I(n, (x0 + k0 - 1) / 2, (x1 + k1 - 1) / 2, ci) * K(2 - k0, 2 - k1, co, ci);
  Program program("defract_long", {O});
  exec::Binder(program).compile()->run();
}

TEST(CppEdsl, DupOut) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20});
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {20, 30});
  auto C = Placeholder(PLAIDML_DATA_FLOAT32, {30, 40});
  auto R = Dot(Dot(A, B), C);
  Program program("dup_out", {R, R, R});
  exec::Binder(program).compile()->run();
}

TEST(CppEdsl, Select) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20});
  auto O = select(I == 0, Tensor{0}, Tensor{1});
  Program program("select", {O});
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  _X0[_X0_0, _X0_1]
) -> (
  _X5
) {
  _X1 = 0;
  _X2 = cmp_eq(_X0, _X1);
  _X3 = 0;
  _X4 = 1;
  _X5 = cond(_X2, _X3, _X4);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
!i32 = type tensor<!eltwise.i32>
module {
  func @select(%arg0: tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.i32> {
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> !i32
    %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !i32
    %0 = "eltwise.cmp_eq"(%arg0, %c0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, !i32) -> tensor<10x20x!eltwise.bool>
    %1 = "eltwise.select"(%0, %c0, %c1) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.bool>, !i32, !i32) -> tensor<10x20x!eltwise.i32>
    return %1 : tensor<10x20x!eltwise.i32>
  }
}
)#"));
#endif
  exec::Binder(program).compile()->run();
}

TEST(CppEdsl, Shape) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20});
  auto O = shape(I);
  Program program("shape", {O});
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  _X0[_X0_0, _X0_1]
) -> (
  _X1
) {
  _X1 = shape(_X0);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
module {
  func @shape(%arg0: tensor<10x20x!eltwise.fp32>) -> tensor<2x!eltwise.i32> {
    %0 = "tile.shape"(%arg0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>) -> tensor<2x!eltwise.i32>
    return %0 : tensor<2x!eltwise.i32>
  }
}
)#"));
#endif
  exec::Binder binder(program);
  binder.compile()->run();
  IVLOG(1, "output: " << O.as_ptr());
  auto view = binder.output(O).mmap_current();
  auto data = reinterpret_cast<const int32_t*>(view.data());
  ASSERT_THAT(view.size(), sizeof(int32_t) * 2);
  EXPECT_THAT(data[0], 10);
  EXPECT_THAT(data[1], 20);
}

TEST(CppEdsl, Prng) {
  auto S = Placeholder(PLAIDML_DATA_UINT32, {3, 2048});
  auto O = prng(S, {2, 3, 4, 5});
  Program program("prng", {O});
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  _X0[_X0_0, _X0_1]
) -> (
  _X6,
  _X7
) {
  _X1 = 2;
  _X2 = 3;
  _X3 = 4;
  _X4 = 5;
  _X5 = prng_step(_X0, _X1, _X2, _X3, _X4);
  _X6 = prng_state(_X5);
  _X7 = prng_value(_X5);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(
!i32 = type tensor<!eltwise.i32>
module {
  func @prng(%arg0: tensor<3x2048x!eltwise.u32>) -> (tensor<2x3x4x5x!eltwise.fp32>, tensor<3x2048x!eltwise.u32>) {
    %c5 = "eltwise.sconst"() {value = 5 : i64} : () -> !i32
    %c4 = "eltwise.sconst"() {value = 4 : i64} : () -> !i32
    %c3 = "eltwise.sconst"() {value = 3 : i64} : () -> !i32
    %c2 = "eltwise.sconst"() {value = 2 : i64} : () -> !i32
    %result, %new_state = "tile.prng"(%arg0, %c2, %c3, %c4, %c5) : (tensor<3x2048x!eltwise.u32>, !i32, !i32, !i32, !i32) -> (tensor<2x3x4x5x!eltwise.fp32>, tensor<3x2048x!eltwise.u32>)
    return %result, %new_state : tensor<2x3x4x5x!eltwise.fp32>, tensor<3x2048x!eltwise.u32>
  }
}
)#"));
#endif
  exec::Binder(program).compile()->run();
}

}  // namespace
}  // namespace plaidml::edsl
