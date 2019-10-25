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

TEST(Op, all) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  auto all = op::all(I);
  IVLOG(1, "all done");
  Program program("all", {all});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3]
) -> (
  _X7
) {
  _X0 = 0;
  _X1 = cmp_eq(I, _X0);
  _X2 = 0;
  _X3 = 1;
  _X4 = cond(_X1, _X2, _X3);
  _X5[] = *(_X4[x0, x1, x2, x3]);
  _X6 = 8;
  _X7 = as_uint(_X5, _X6);
}
)"));
}

TEST(Op, any) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  auto any = op::any(I);
  IVLOG(1, "any done");
  Program program("any", {any});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3]
) -> (
  _X12
) {
  _X0 = 0;
  _X1 = cmp_eq(I, _X0);
  _X2 = 0;
  _X3 = 1;
  _X4 = cond(_X1, _X2, _X3);
  _X5[] = +(_X4[x0, x1, x2, x3]);
  _X6 = 0;
  _X7 = cmp_eq(_X5, _X6);
  _X8 = 0;
  _X9 = 1;
  _X10 = cond(_X7, _X8, _X9);
  _X11 = 8;
  _X12 = as_uint(_X10, _X11);
}
)"));
}

TEST(Op, argmax) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  auto argmax = op::argmax(I);
  IVLOG(1, "argmax done");
  Program program("argmax", {argmax});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3]
) -> (
  _X7
) {
  _X0[] = >(I[x0, x1, x2, x3]);
  _X1 = 1;
  _X2[x0, x1, x2, x3 : 1, 224, 224, 3] = =(_X1[]);
  _X3 = 0;
  _X4 = index(_X2, _X3);
  _X5[] = >(I[x0, x1, x2, x3] == _X0[] ? _X4[x0, x1, x2, x3]);
  _X6 = 32;
  _X7 = as_uint(_X5, _X6);
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
      {1, 3} /* axes to squeeze*/);
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

TEST(Op, Tile) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  auto t = op::tile(  //
      A,              // tensor to tile
      {5, 4} /* tiling factors*/);
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
