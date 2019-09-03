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

}  // namespace
}  // namespace op
}  // namespace plaidml
