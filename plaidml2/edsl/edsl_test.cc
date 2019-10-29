// Copyright 2019 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "base/util/env.h"
#include "base/util/logging.h"
#include "plaidml2/edsl/autodiff.h"
#include "plaidml2/edsl/edsl.h"
#include "plaidml2/exec/exec.h"

using ::testing::Eq;

namespace plaidml::edsl {

bool operator==(const Program& lhs, const std::string& rhs) {  //
  return lhs.str() == rhs;
}

namespace {

Tensor Dot(const Tensor& X, const Tensor& Y) {
  TensorDim I, J, K;
  TensorIndex i, j, k;
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
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {1, 784});
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {784, 512});
  Program program("dot", {Dot(A, B)});
  exec::Executable::compile(program, {A, B})->run();
}

TEST(CppEdsl, DoubleDot) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20});
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {20, 30});
  auto C = Placeholder(PLAIDML_DATA_FLOAT32, {30, 40});
  Program program("double_dot", {Dot(Dot(A, B), C)});
  exec::Executable::compile(program, {A, B, C})->run();
}

TEST(CppEdsl, EltwiseAdd) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20});
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20});
  auto C = A + B;
  Program program("eltwise_add", {C});
  exec::Executable::compile(program, {A, B})->run();
}

TEST(CppEdsl, Relu) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20});
  Program program("relu", {Relu(A)});
  exec::Executable::compile(program, {A})->run();
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
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type tensor<!eltwise.fp32>
module {
  func @mnist_mlp(%arg0: tensor<10x!eltwise.fp32>, %arg1: tensor<512x!eltwise.fp32>, %arg2: tensor<512x!eltwise.fp32>, %arg3: tensor<1x784x!eltwise.fp32>, %arg4: tensor<784x512x!eltwise.fp32>, %arg5: tensor<512x512x!eltwise.fp32>, %arg6: tensor<512x10x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32> {
    %c512 = "tile.affine_const"() {value = 512 : i64} : () -> index
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f32} : () -> !fp32
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> index
    %c0 = "tile.affine_const"() {value = 0 : i64} : () -> index
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> index
    %0 = "tile.domain"() ( {
    ^bb0(%arg7: index, %arg8: index, %arg9: index):	// no predecessors
      %15 = "tile.src_idx_map"(%arg3, %arg8, %arg7) : (tensor<1x784x!eltwise.fp32>, index, index) -> !tile.imap
      %16 = "tile.src_idx_map"(%arg4, %arg7, %arg9) : (tensor<784x512x!eltwise.fp32>, index, index) -> !tile.imap
      %17 = "tile.sink_idx_map"(%arg8, %arg9) : (index, index) -> !tile.imap
      %18 = "tile.size_map"(%c1, %c512) : (index, index) -> !tile.smap
      "tile.+(x*y)"(%18, %15, %16, %17) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x512x!eltwise.fp32>
    %1 = "eltwise.add"(%0, %arg2) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.fp32>, tensor<512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32>
    %2 = "eltwise.cmp_lt"(%1, %cst) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.fp32>, !fp32) -> tensor<1x512x!eltwise.bool>
    %3 = "eltwise.select"(%2, %cst, %1) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.bool>, !fp32, tensor<1x512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32>
    %4 = "tile.domain"() ( {
    ^bb0(%arg7: index, %arg8: index, %arg9: index):	// no predecessors
      %15 = "tile.src_idx_map"(%3, %arg8, %arg7) : (tensor<1x512x!eltwise.fp32>, index, index) -> !tile.imap
      %16 = "tile.src_idx_map"(%arg5, %arg7, %arg9) : (tensor<512x512x!eltwise.fp32>, index, index) -> !tile.imap
      %17 = "tile.sink_idx_map"(%arg8, %arg9) : (index, index) -> !tile.imap
      %18 = "tile.size_map"(%c1, %c512) : (index, index) -> !tile.smap
      "tile.+(x*y)"(%18, %15, %16, %17) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x512x!eltwise.fp32>
    %5 = "eltwise.add"(%4, %arg1) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.fp32>, tensor<512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32>
    %6 = "eltwise.cmp_lt"(%5, %cst) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.fp32>, !fp32) -> tensor<1x512x!eltwise.bool>
    %7 = "eltwise.select"(%6, %cst, %5) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.bool>, !fp32, tensor<1x512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32>
    %8 = "tile.domain"() ( {
    ^bb0(%arg7: index, %arg8: index, %arg9: index):	// no predecessors
      %15 = "tile.src_idx_map"(%7, %arg8, %arg7) : (tensor<1x512x!eltwise.fp32>, index, index) -> !tile.imap
      %16 = "tile.src_idx_map"(%arg6, %arg7, %arg9) : (tensor<512x10x!eltwise.fp32>, index, index) -> !tile.imap
      %17 = "tile.sink_idx_map"(%arg8, %arg9) : (index, index) -> !tile.imap
      %18 = "tile.size_map"(%c1, %c10) : (index, index) -> !tile.smap
      "tile.+(x*y)"(%18, %15, %16, %17) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x10x!eltwise.fp32>
    %9 = "eltwise.add"(%8, %arg0) {type = !eltwise.fp32} : (tensor<1x10x!eltwise.fp32>, tensor<10x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32>
    %10 = "tile.domain"() ( {
    ^bb0(%arg7: index, %arg8: index):	// no predecessors
      %15 = "tile.src_idx_map"(%9, %arg8, %arg7) : (tensor<1x10x!eltwise.fp32>, index, index) -> !tile.imap
      %16 = "tile.sink_idx_map"(%arg8, %c0) : (index, index) -> !tile.imap
      %17 = "tile.size_map"(%c1, %c1) : (index, index) -> !tile.smap
      "tile.>(x)"(%17, %15, %16) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<1x1x!eltwise.fp32>
    %11 = "eltwise.sub"(%9, %10) {type = !eltwise.fp32} : (tensor<1x10x!eltwise.fp32>, tensor<1x1x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32>
    %12 = "eltwise.exp"(%11) {type = !eltwise.fp32} : (tensor<1x10x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32>
    %13 = "tile.domain"() ( {
    ^bb0(%arg7: index, %arg8: index):	// no predecessors
      %15 = "tile.src_idx_map"(%12, %arg8, %arg7) : (tensor<1x10x!eltwise.fp32>, index, index) -> !tile.imap
      %16 = "tile.sink_idx_map"(%arg8, %c0) : (index, index) -> !tile.imap
      %17 = "tile.size_map"(%c1, %c1) : (index, index) -> !tile.smap
      "tile.+(x)"(%17, %15, %16) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<1x1x!eltwise.fp32>
    %14 = "eltwise.div"(%12, %13) {type = !eltwise.fp32} : (tensor<1x10x!eltwise.fp32>, tensor<1x1x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32>
    return %14 : tensor<1x10x!eltwise.fp32>
  }
}
)#"));
  std::vector<Tensor> inputs{input, kernel1, bias1, kernel2, bias2, kernel3, bias3};
  exec::Executable::compile(program, inputs)->run();
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
  // This currently crashes when combined with the padding pass
  // exec::Executable::compile(program, {I, K})->run();
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
  // This currently crashes when combined with the padding pass
  std::vector<Tensor> inputs{input, kernel1, bias1, kernel2, bias2, kernel3, bias3, kernel4, bias4};
  exec::Executable::compile(program, inputs)->run();
}

Tensor Normalize(const Tensor& X) {
  auto XSqr = X * X;
  auto X_MS = TensorOutput();
  std::vector<TensorIndex> idxs(X.shape().ndims());
  X_MS() += XSqr(idxs);
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

TEST(CppEdsl, LarsMomentum4d) {
  auto X_shape = LogicalShape(PLAIDML_DATA_FLOAT32, {4, 7, 3, 9});
  auto LR_shape = LogicalShape(PLAIDML_DATA_FLOAT32, {});
  auto X = Placeholder(X_shape);
  auto Grad = Placeholder(X_shape);
  auto Veloc = Placeholder(X_shape);
  auto LR = Placeholder(LR_shape);
  auto R = LarsMomentum(X, Grad, Veloc, LR, 1. / 1024., 1. / 2048., 1. / 8.);
  Program program("lars_momentum4d", {std::get<0>(R), std::get<1>(R)});
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type tensor<!eltwise.fp32>
module {
  func @lars_momentum4d(%arg0: tensor<4x7x3x9x!eltwise.fp32>, %arg1: tensor<4x7x3x9x!eltwise.fp32>, %arg2: !fp32, %arg3: tensor<4x7x3x9x!eltwise.fp32>) -> (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) {
    %cst = "eltwise.sconst"() {value = 4.8828125E-4 : f32} : () -> !fp32
    %cst_0 = "eltwise.sconst"() {value = 9.765625E-4 : f32} : () -> !fp32
    %cst_1 = "eltwise.sconst"() {value = 1.250000e-01 : f32} : () -> !fp32
    %0 = "eltwise.mul"(%arg0, %cst) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, !fp32) -> tensor<4x7x3x9x!eltwise.fp32>
    %1 = "eltwise.add"(%arg1, %0) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %2 = "eltwise.mul"(%arg0, %arg0) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %3 = "tile.domain"() ( {
    ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):	// no predecessors
      %17 = "tile.src_idx_map"(%2, %arg7, %arg6, %arg5, %arg4) : (tensor<4x7x3x9x!eltwise.fp32>, index, index, index, index) -> !tile.imap
      %18 = "tile.sink_idx_map"() : () -> !tile.imap
      %19 = "tile.size_map"() : () -> !tile.smap
      "tile.+(x)"(%19, %17, %18) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> !fp32
    %4 = "eltwise.sqrt"(%3) {type = !eltwise.fp32} : (!fp32) -> !fp32
    %5 = "eltwise.mul"(%4, %cst) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %6 = "eltwise.mul"(%arg1, %arg1) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %7 = "tile.domain"() ( {
    ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):	// no predecessors
      %17 = "tile.src_idx_map"(%6, %arg7, %arg6, %arg5, %arg4) : (tensor<4x7x3x9x!eltwise.fp32>, index, index, index, index) -> !tile.imap
      %18 = "tile.sink_idx_map"() : () -> !tile.imap
      %19 = "tile.size_map"() : () -> !tile.smap
      "tile.+(x)"(%19, %17, %18) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> !fp32
    %8 = "eltwise.sqrt"(%7) {type = !eltwise.fp32} : (!fp32) -> !fp32
    %9 = "eltwise.add"(%8, %5) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %10 = "eltwise.mul"(%arg2, %cst_0) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %11 = "eltwise.mul"(%10, %4) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %12 = "eltwise.div"(%11, %9) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %13 = "eltwise.mul"(%12, %1) {type = !eltwise.fp32} : (!fp32, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %14 = "eltwise.mul"(%arg3, %cst_1) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, !fp32) -> tensor<4x7x3x9x!eltwise.fp32>
    %15 = "eltwise.add"(%14, %13) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %16 = "eltwise.sub"(%arg0, %15) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    return %16, %15 : tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>
  }
}
)#"));
  std::vector<Tensor> inputs{X, Grad, Veloc, LR};
  exec::Executable::compile(program, inputs)->run();
}

// TODO: no_defract
TEST(CppEdsl, DISABLED_RepeatElements) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 10, 10});
  TensorDim N0, N1, N2;
  TensorIndex n0, n1, n2, k;
  I.bind_dims(N0, N1, N2);
  auto O = TensorOutput(N0, 3 * N1, N2);
  O(n0, 3 * n1 + k, n2) = I(n0, n1, n2);
  O.add_constraint(k < 3);
  O.no_defract();
  Program program("repeat_elts", {O});
  EXPECT_THAT(program, Eq(R"(function (
  _X0[_X0_0, _X0_1, _X0_2]
) -> (
  _X1
) {
  _X1[x0, 3*x1 + x3, x2 : 10, 30, 10] = =(_X0[x0, x1, x2]), x3 < 3 no_defract;
}
)"));
  exec::Executable::compile(program, {I})->run();
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
  EXPECT_THAT(program, Eq(R"#(

module {
  func @use_default(%arg0: tensor<1x10x10x!eltwise.fp32>, %arg1: tensor<1x7x10x10x!eltwise.fp32>) -> tensor<1x7x10x10x!eltwise.fp32> {
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> index
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> index
    %c7 = "tile.affine_const"() {value = 7 : i64} : () -> index
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> index
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: index, %arg3: index, %arg4: index):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2) : (tensor<1x10x10x!eltwise.fp32>, index, index, index) -> !tile.imap
      %2 = "tile.sink_idx_map"(%arg4, %c3, %arg3, %arg2) : (index, index, index, index) -> !tile.imap
      %3 = "tile.size_map"(%c1, %c7, %c10, %c10) : (index, index, index, index) -> !tile.smap
      "tile.=(x)"(%3, %1, %2, %arg1) : (!tile.smap, !tile.imap, !tile.imap, tensor<1x7x10x10x!eltwise.fp32>) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x7x10x10x!eltwise.fp32>
    return %0 : tensor<1x7x10x10x!eltwise.fp32>
  }
}
)#"));
  exec::Executable::compile(program, {P, I})->run();
}

Tensor ArgMax(const Tensor& I) {
  TensorDim X0, X1, X2;
  TensorIndex x0, x1, x2;
  I.bind_dims(X0, X1, X2);
  auto Max = TensorOutput(X0, X2);
  Max(x0, x2) >= I(x0, x1, x2);
  auto One = Placeholder(I.shape().dtype(), {});
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
  EXPECT_THAT(program, Eq(R"#(

!i32 = type tensor<!eltwise.i32>
!fp32 = type tensor<!eltwise.fp32>
module {
  func @arg_max(%arg0: tensor<1x10x10x!eltwise.fp32>, %arg1: !fp32) -> tensor<1x10x!eltwise.u32> {
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> index
    %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !i32
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> index
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: index, %arg3: index, %arg4: index):	// no predecessors
      %5 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2) : (tensor<1x10x10x!eltwise.fp32>, index, index, index) -> !tile.imap
      %6 = "tile.sink_idx_map"(%arg4, %arg2) : (index, index) -> !tile.imap
      %7 = "tile.size_map"(%c1, %c10) : (index, index) -> !tile.smap
      "tile.>(x)"(%7, %5, %6) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x10x!eltwise.fp32>
    %1 = "tile.domain"() ( {
    ^bb0(%arg2: index):	// no predecessors
      %5 = "tile.src_idx_map"(%arg1) : (!fp32) -> !tile.imap
      %6 = "tile.sink_idx_map"(%arg2) : (index) -> !tile.imap
      %7 = "tile.size_map"(%c10) : (index) -> !tile.smap
      "tile.=(x)"(%7, %5, %6) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0"]} : () -> tensor<10x!eltwise.fp32>
    %2 = "tile.index"(%1, %c0) : (tensor<10x!eltwise.fp32>, !i32) -> tensor<10x!eltwise.i32>
    %3 = "tile.domain"() ( {
    ^bb0(%arg2: index, %arg3: index, %arg4: index):	// no predecessors
      %5 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2) : (tensor<1x10x10x!eltwise.fp32>, index, index, index) -> !tile.imap
      %6 = "tile.src_idx_map"(%0, %arg4, %arg2) : (tensor<1x10x!eltwise.fp32>, index, index) -> !tile.imap
      %7 = "tile.src_idx_map"(%2, %arg3) : (tensor<10x!eltwise.i32>, index) -> !tile.imap
      %8 = "tile.sink_idx_map"(%arg4, %arg2) : (index, index) -> !tile.imap
      %9 = "tile.size_map"(%c1, %c10) : (index, index) -> !tile.smap
      "tile.>(x==y?z)"(%9, %5, %6, %7, %8) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x10x!eltwise.fp32>
    %4 = "eltwise.as_uint"(%3) : (tensor<1x10x!eltwise.fp32>) -> tensor<1x10x!eltwise.u32>
    return %4 : tensor<1x10x!eltwise.u32>
  }
}
)#"));
  // TODO: cpu backend is missing cast ops (as_uint)
  // exec::Executable::compile(program, {I})->run();
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
  O.no_defract();
  return O;
}

// TODO: no_defract
TEST(CppEdsl, DISABLED_Winograd) {
  const std::int64_t N = 1, X = 224, Y = 224, CI = 3, S = 3, CO = 32, BI = 32, BO = BI - CI + 1;
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {N, X, Y, CI});
  auto K = Placeholder(PLAIDML_DATA_FLOAT32, {S, S, CI, CO});
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {BI, BO});
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {BI, BI});
  auto G = Placeholder(PLAIDML_DATA_FLOAT32, {BI, S});
  auto W = Winograd(I, K, A, B, G);
  Program program("winograd", {W});
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
  // This currently crashes when combined with the padding pass
  // exec::Executable::compile(program, {I, K, A, B, G})->run();
}

TEST(CppEdsl, UniqueNames) {
  LogicalShape shape(PLAIDML_DATA_FLOAT32, {});
  auto A = Placeholder(shape, "A");
  auto B = Placeholder(shape, "B");
  auto C0 = Placeholder(shape, "C");
  auto C1 = Placeholder(shape, "C");
  Program program("unique_names", {A + B + C0 + C1});
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type tensor<!eltwise.fp32>
module {
  func @unique_names(%arg0: !fp32 {tile.name = "C"}, %arg1: !fp32 {tile.name = "C"}, %arg2: !fp32 {tile.name = "B"}, %arg3: !fp32 {tile.name = "A"}) -> !fp32 {
    %0 = "eltwise.add"(%arg3, %arg2) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %1 = "eltwise.add"(%0, %arg1) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %2 = "eltwise.add"(%1, %arg0) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    return %2 : !fp32
  }
}
)#"));
  exec::Executable::compile(program, {A, B, C0, C1})->run();
}

TEST(CppEdsl, GlobalMin) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 10, 10}, "I");
  TensorIndex i, j, k;
  auto O_Neg = TensorOutput();
  auto Neg = -I;
  O_Neg() >= Neg(i, j, k);
  auto O = -O_Neg;
  Program program("global_min", {O});
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type tensor<!eltwise.fp32>
module {
  func @global_min(%arg0: tensor<10x10x10x!eltwise.fp32> {tile.name = "I"}) -> !fp32 {
    %0 = "eltwise.neg"(%arg0) {type = !eltwise.fp32} : (tensor<10x10x10x!eltwise.fp32>) -> tensor<10x10x10x!eltwise.fp32>
    %1 = "tile.domain"() ( {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):	// no predecessors
      %3 = "tile.src_idx_map"(%0, %arg3, %arg2, %arg1) : (tensor<10x10x10x!eltwise.fp32>, index, index, index) -> !tile.imap
      %4 = "tile.sink_idx_map"() : () -> !tile.imap
      %5 = "tile.size_map"() : () -> !tile.smap
      "tile.>(x)"(%5, %3, %4) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> !fp32
    %2 = "eltwise.neg"(%1) {type = !eltwise.fp32} : (!fp32) -> !fp32
    return %2 : !fp32
  }
}
)#"));
  exec::Executable::compile(program, {I})->run();
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
  EXPECT_THAT(program, Eq(R"#(

module {
  func @cumsum(%arg0: tensor<10x!eltwise.fp32> {tile.name = "I"}) -> tensor<10x!eltwise.fp32> {
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> index
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: index, %arg2: index):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg1) : (tensor<10x!eltwise.fp32>, index) -> !tile.imap
      %2 = "tile.sink_idx_map"(%arg2) : (index) -> !tile.imap
      %3 = "tile.size_map"(%c10) : (index) -> !tile.smap
      %4 = "tile.affine_sub"(%arg2, %arg1) : (index, index) -> index
      "tile.constraint"(%4, %c10) ( {
        "tile.+(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
      }) : (index, index) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<10x!eltwise.fp32>
    return %0 : tensor<10x!eltwise.fp32>
  }
}
)#"));
  exec::Executable::compile(program, {I})->run();
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
  EXPECT_THAT(program, Eq(R"#(

module {
  func @complex_conv_2d(%arg0: tensor<1x224x224x3x3x!eltwise.fp32>, %arg1: tensor<3x3x3x3x32x!eltwise.fp32>) -> tensor<1x112x112x3x32x!eltwise.fp32> {
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> index
    %c112 = "tile.affine_const"() {value = 112 : i64} : () -> index
    %c32 = "tile.affine_const"() {value = 32 : i64} : () -> index
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> index
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> index
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index):	// no predecessors
      %1 = "tile.affine_mul"(%arg4, %c3) : (index, index) -> index
      %2 = "tile.affine_mul"(%arg5, %c2) : (index, index) -> index
      %3 = "tile.affine_add"(%2, %1) : (index, index) -> index
      %4 = "tile.affine_sub"(%3, %c2) : (index, index) -> index
      %5 = "tile.affine_mul"(%arg6, %c3) : (index, index) -> index
      %6 = "tile.affine_mul"(%arg7, %c2) : (index, index) -> index
      %7 = "tile.affine_add"(%6, %5) : (index, index) -> index
      %8 = "tile.affine_sub"(%7, %c2) : (index, index) -> index
      %9 = "tile.src_idx_map"(%arg0, %arg8, %8, %4, %arg3, %arg2) : (tensor<1x224x224x3x3x!eltwise.fp32>, index, index, index, index, index) -> !tile.imap
      %10 = "tile.src_idx_map"(%arg1, %arg6, %arg4, %arg3, %arg2, %arg9) : (tensor<3x3x3x3x32x!eltwise.fp32>, index, index, index, index, index) -> !tile.imap
      %11 = "tile.sink_idx_map"(%arg8, %arg7, %arg5, %arg3, %arg9) : (index, index, index, index, index) -> !tile.imap
      %12 = "tile.size_map"(%c1, %c112, %c112, %c3, %c32) : (index, index, index, index, index) -> !tile.smap
      "tile.+(x*y)"(%12, %9, %10, %11) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]} : () -> tensor<1x112x112x3x32x!eltwise.fp32>
    return %0 : tensor<1x112x112x3x32x!eltwise.fp32>
  }
}
)#"));
  exec::Executable::compile(program, {I, K})->run();
}

TEST(CppEdsl, Reciprocal) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10}, "A");
  Program program("reciprocal", {1 / A});
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
  exec::Executable::compile(program, {A})->run();
}

TEST(CppEdsl, DISABLED_GradientDot) {
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
  _X2[x1, x2 : 100, 100] = +(A[x0, x1] * _X1[x0, x2]);
  _X3[x0, x2 : 100, 100] = +(_X1[x0, x1] * B[x2, x1]);
}
)"));
  exec::Executable::compile(program, {A, B})->run();
}

Tensor Max2Da0(const Tensor& A) {
  TensorDim M, N;
  A.bind_dims(M, N);
  TensorIndex m("m"), n("n");
  auto O = NamedTensorOutput("O", N);
  O(n) >= A(m, n);
  // O(n) += A(m, n);
  return O;
}

TEST(CppEdsl, DISABLED_GradientMultiDot) {
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
  _X0[x0, x2 : 100, 100] = +(A[x0, x1] * B[x1, x2]);
  _X1[x0, x2 : 100, 100] = +(A[x0, x1] * _X0[x1, x2]);
  O[n : 100] = >(_X1[m, n]);
  _X2 = 1.000000;
  _X3[x0 : 100] = +(_X2[]);
  _X4[m, n : 100, 100] = +(_X1[m, n] == O[n] ? _X3[n]);
  _X5[x1, x2 : 100, 100] = +(A[x0, x1] * _X4[x0, x2]);
  _X6[x1, x2 : 100, 100] = +(A[x0, x1] * _X5[x0, x2]);
  _X7[x0, x2 : 100, 100] = +(_X4[x0, x1] * _X0[x2, x1]);
  _X8[x0, x2 : 100, 100] = +(_X5[x0, x1] * B[x2, x1]);
  _X9 = add(_X7, _X8);
}
)"));
  exec::Executable::compile(program, {A, B})->run();
}

TEST(CppEdsl, DISABLED_GradientDotSqrt) {
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
  _X3[x0, x2 : 100, 100] = +(A[x0, x1] * B[x1, x2]);
  _X4 = sqrt(_X3);
  _X5 = mul(_X2, _X4);
  _X6 = div(_X1, _X5);
  _X7[x1, x2 : 100, 100] = +(A[x0, x1] * _X6[x0, x2]);
  _X8[x0, x2 : 100, 100] = +(_X6[x0, x1] * B[x2, x1]);
}
)"));
  exec::Executable::compile(program, {A, B})->run();
}

TEST(CppEdsl, DefractLong) {
  std::vector<int64_t> input_shape{1, 3, 3, 1};
  std::vector<int64_t> output_shape{1, 5, 5, 1};
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, input_shape, "I");
  auto K = Placeholder(PLAIDML_DATA_FLOAT32, input_shape, "K");
  auto O = TensorOutput(output_shape);
  TensorIndex n, x0, x1, k0, k1, co, ci;
  O(n, x0, x1, co) += I(n, (x0 + k0 - 1) / 2, (x1 + k1 - 1) / 2, ci) * K(2 - k0, 2 - k1, co, ci);
  Program program("defract_long", {O});
  exec::Executable::compile(program, {I, K})->run();
}

TEST(CppEdsl, DupOut) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20});
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {20, 30});
  auto C = Placeholder(PLAIDML_DATA_FLOAT32, {30, 40});
  auto R = Dot(Dot(A, B), C);
  Program program("dup_out", {R, R, R});
  exec::Executable::compile(program, {A, B, C})->run();
}

}  // namespace
}  // namespace plaidml::edsl
