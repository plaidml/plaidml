// Copyright 2019 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "base/util/logging.h"
#include "plaidml2/op/op.h"

using ::testing::Eq;

using namespace plaidml::edsl;  // NOLINT

namespace plaidml {
namespace edsl {

bool operator==(const Program& lhs, const std::string& rhs) {  //
  return lhs.str() == rhs;
}

}  // namespace edsl

namespace op {
namespace {

class Environment : public ::testing::Environment {
  void SetUp() override {  //
    plaidml::op::init();
  }
};

[[gnu::unused]] auto init = []() {  //
  ::testing::AddGlobalTestEnvironment(new Environment);
  return 0;
}();

TEST(Op, abs) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  auto abs = op::abs(I);
  IVLOG(1, "Abs done");
  Program program("abs", {abs});
  IVLOG(1, program);

  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3]
) -> (
  _X3
) {
  _X0 = 0.000000;
  _X1 = cmp_lt(I, _X0);
  _X2 = neg(I);
  _X3 = cond(_X1, _X2, I);
}
)"));
}

TEST(Op, Convolution) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  auto K = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "K");
  auto conv = op::convolution(  //
      I,                        // I_or_O
      K,                        // F_or_O
      {2, 2},                   // strides
      {1, 1},                   // dilations
      {1, 1},                   // data_dilations
      {},                       // filter_shape
      1,                        // groups
      "explicit",               // autopad_mode
      {3, 3},                   // manual_padding
      "nxc",                    // input_layout
      "xck",                    // filter_layout
      "none",                   // group_layout
      false,                    // winograd_allowed
      "",                       // name
      "ungrouped",              // autogroup_mode
      "none",                   // deriv_mode
      {});                      // result_shape
  IVLOG(1, "Conv done");
  Program program("convolution", {conv});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3],
  K[K_0, K_1, K_2, K_3]
) -> (
  conv
) {
  conv[n, x0, x1, co : 1, 112, 112, 64] = +(I[n, -3 + k0 + 2*x0, -3 + k1 + 2*x1, ci] * K[k0, k1, ci, co]);
}
)"));
}

TEST(Op, Mean) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("mean", {op::mean(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X2
) {
  _X0[] = +(A[x0, x1]);
  _X1 = 200;
  _X2 = div(_X0, _X1);
}
)"));
}

TEST(Op, Prod) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("prod", {op::prod(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X0
) {
  _X0[] = *(A[x0, x1]);
}
)"));
}

TEST(Op, Relu) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "I");
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  auto M = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "M");
  auto r = op::relu(I).alpha(A).max_value(M).threshold(0.05).create();
  Program program("relu", {r});
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1],
  A[A_0, A_1],
  M[M_0, M_1]
) -> (
  _X7
) {
  _X0 = 0.050000;
  _X1 = cmp_lt(I, _X0);
  _X2 = 0.050000;
  _X3 = sub(I, _X2);
  _X4 = mul(A, _X3);
  _X5 = cond(_X1, _X4, I);
  _X6 = cmp_lt(_X5, M);
  _X7 = cond(_X6, _X5, M);
}
)"));
}

TEST(Op, ReluNoAlpha) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "I");
  auto M = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "M");
  auto r = op::relu(I).max_value(M).threshold(0.05).create();
  Program program("relu", {r});
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1],
  M[M_0, M_1]
) -> (
  _X8
) {
  _X0 = 0.050000;
  _X1 = cmp_lt(I, _X0);
  _X2 = 0.000000;
  _X3 = 0.050000;
  _X4 = sub(I, _X3);
  _X5 = mul(_X2, _X4);
  _X6 = cond(_X1, _X5, I);
  _X7 = cmp_lt(_X6, M);
  _X8 = cond(_X7, _X6, M);
}
)"));
}

TEST(Op, ReluNoMaxValue) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "I");
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  auto r = op::relu(I).alpha(A).threshold(0.05).create();
  Program program("relu", {r});
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1],
  A[A_0, A_1]
) -> (
  _X5
) {
  _X0 = 0.050000;
  _X1 = cmp_lt(I, _X0);
  _X2 = 0.050000;
  _X3 = sub(I, _X2);
  _X4 = mul(A, _X3);
  _X5 = cond(_X1, _X4, I);
}
)"));
}

TEST(Op, ReluOnlyThreshold) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "I");
  auto r = op::relu(I).threshold(0.05).create();
  Program program("relu", {r});
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1]
) -> (
  _X6
) {
  _X0 = 0.050000;
  _X1 = cmp_lt(I, _X0);
  _X2 = 0.000000;
  _X3 = 0.050000;
  _X4 = sub(I, _X3);
  _X5 = mul(_X2, _X4);
  _X6 = cond(_X1, _X5, I);
}
)"));
}

TEST(Op, ReluNoParams) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "I");
  auto r = op::relu(I).create();
  Program program("relu", {r});
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1]
) -> (
  _X4
) {
  _X0 = 0.000000;
  _X1 = cmp_lt(I, _X0);
  _X2 = 0.000000;
  _X3 = mul(_X2, I);
  _X4 = cond(_X1, _X3, I);
}
)"));
}

TEST(Op, Repeat) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {32, 1, 4, 1}, "A");
  auto t = op::repeat(  //
      A,                // tensor to repeat
      3,                // number of repeats
      2);               // axis to repeat
  Program program("repeat", {t});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1, A_2, A_3]
) -> (
  _X0
) {
  _X0[x0, x1, 3*x2 + x4, x3 : 32, 1, 12, 1] = =(A[x0, x1, x2, x3]), x4 < 3;
}
)"));
}

TEST(Op, Reshape) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  TensorDim I, J;
  A.bind_dims(I, J);
  Program program("reshape", {op::reshape(A, make_tuple(J, I))});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X2
) {
  _X0 = 20;
  _X1 = 10;
  _X2 = reshape(A, _X0, _X1);
}
)"));
}

TEST(Op, Sigmoid) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10}, "A");
  Program program("sigmoid", {op::sigmoid(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0]
) -> (
  _X7
) {
  _X0 = ident(A);
  _X1 = 1.000000;
  _X2 = 1.000000;
  _X3 = neg(_X0);
  _X4 = exp(_X3);
  _X5 = add(_X2, _X4);
  _X6 = div(_X1, _X5);
  _X7 = ident(_X6);
}
)"));
}

TEST(Op, Slice) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  auto t = op::slice(  //
      A,               // tensor to perform spatial padding on
      {2, 10});        // slices
  Program program("slice", {t});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X0
) {
  _X0[] = =(A[2, 10]);
}
)"));
}

TEST(Op, Softmax) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("softmax", {op::softmax(A, 1)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X6
) {
  _X0 = ident(A);
  _X1[x0, 0 : 10, 1] = >(_X0[x0, x1]);
  _X2 = sub(_X0, _X1);
  _X3 = exp(_X2);
  _X4[x0, 0 : 10, 1] = +(_X3[x0, x1]);
  _X5 = div(_X3, _X4);
  _X6 = ident(_X5);
}
)"));
}

TEST(Op, SpatialPadding) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {64, 4, 32, 32}, "A");
  auto t = op::spatial_padding(  //
      A,                         // tensor to perform spatial padding on
      {1, 3},                    // low pads
      {3, 3},                    // high pads
      "nchw");                   // data layout
  Program program("spatial_padding", {t});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1, A_2, A_3]
) -> (
  _X0
) {
  _X0[n, c, 1 + x0, 3 + x1 : 64, 4, 36, 38] = =(A[n, c, x0, x1]);
}
)"));
}

TEST(Op, Square) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10}, "A");
  Program program("square", {op::square(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0]
) -> (
  _X0
) {
  _X0 = mul(A, A);
}
)"));
}

TEST(Op, Sum) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("sum", {op::sum(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X0
) {
  _X0[] = +(A[x0, x1]);
}
)"));
}

TEST(Op, Squeeze) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {32, 1, 4, 1}, "A");
  auto t = op::squeeze(  //
      A,                 // tensor to squeeze
      {1, 3});           // axes to squeeze
  Program program("squeeze", {t});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1, A_2, A_3]
) -> (
  _X2
) {
  _X0 = 32;
  _X1 = 4;
  _X2 = reshape(A, _X0, _X1);
}
)"));
}

TEST(Op, Tile) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  auto t = op::tile(  //
      A,              // tensor to tile
      {5, 4});        // tiling factors
  Program program("tile", {t});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X0
) {
  _X0[x0 + 10*x2, x1 + 20*x3 : 50, 80] = =(A[x0, x1]) no_defract;
}
)"));
}

TEST(Op, Transpose) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("transpose", {op::transpose(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X0
) {
  _X0[x1, x0 : 20, 10] = =(A[x0, x1]);
}
)"));
}

}  // namespace
}  // namespace op
}  // namespace plaidml
