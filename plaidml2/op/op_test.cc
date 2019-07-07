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

TEST(Op, Convolution) {
  Tensor I("I", LogicalShape(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, NCHW));
  Tensor K("K", LogicalShape(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, HWCK));
  auto conv = op::convolution(I, K, {2, 2}, {3, 3}, {1, 1}, 1);
  EXPECT_THAT(conv.shape().layout(), Eq(NHWC));
  Program program("convolution", {conv});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3],
  K[K_0, K_1, K_2, K_3]
) -> (
  _X0
) {
  _X0[x0, x1, x3, x6 : 1, 112, 112, 64] = +(I[x0, -3 + 2*x1 + x2, -3 + 2*x3 + x4, x5] * K[x2, x4, x5, x6]);
}
)"));
}

TEST(Op, Mean) {
  Tensor A("A", LogicalShape(PLAIDML_DATA_FLOAT32, {10, 20}));
  Program program("mean", {op::mean(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X4
) {
  _X0[] = +(A[x0, x1]);
  _X1 = 10;
  _X2 = 20;
  _X3 = mul(_X1, _X2);
  _X4 = div(_X0, _X3);
}
)"));
}

TEST(Op, Square) {
  Tensor A("A", LogicalShape(PLAIDML_DATA_FLOAT32, {10}));
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
  Tensor A("A", LogicalShape(PLAIDML_DATA_FLOAT32, {10, 20}));
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

}  // namespace
}  // namespace op
}  // namespace plaidml
