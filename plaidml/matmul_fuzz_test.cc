// Copyright 2018 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <tuple>

#include "base/util/logging.h"
#include "plaidml/base/base.h"
#include "plaidml/base/context.h"
#include "plaidml/plaidml++.h"
#include "plaidml/plaidml.h"
#include "testing/plaidml_config.h"

using ::testing::Eq;
using ::testing::IsNull;
using ::testing::Ne;
using ::testing::NotNull;
using ::testing::StrEq;

extern "C" void vai_internal_set_vlog(size_t);

namespace {

namespace plaidml = vertexai::plaidml;

class MatMulTest : public ::testing::TestWithParam<std::tuple<size_t, size_t, size_t>> {
 public:
  static void SetUpTestCase() {
    auto ctx = std::make_shared<vertexai::ctx>();
    auto devices = plaidml::enumerate_devices(ctx, vertexai::testing::PlaidMLConfig());
    dev_ = devices[0].open();
    matmul_ = plaidml::function{"function (B[X,Z], C[Z,Y]) -> (A) { A[x,y : X,Y] = +(B[x,z] * C[z,y]); }"};
  }

 protected:
  static plaidml::device dev_;
  static plaidml::function matmul_;
};

plaidml::device MatMulTest::dev_;
plaidml::function MatMulTest::matmul_;

TEST_P(MatMulTest, Fuzz) {
  vai_clear_status();

  auto ctx = std::make_shared<vertexai::ctx>();

  auto p = GetParam();
  auto x = std::get<0>(p);
  auto z = std::get<1>(p);
  auto y = std::get<2>(p);
  LOG(INFO) << "Testing [" << x << ", " << z << "] * [" << z << ", " << y << "] -> [" << x << ", " << y << "]";

  // Setup the lhs tensor of data
  plaidml::tensor<float> b = dev_.allocate(plaidml::shape<float>(ctx, {x, z}));
  {
    plaidml::mapping<float> data = b.map(plaidml::map_for_write);
    for (size_t i = 0; i < x; i++) {
      for (size_t j = 0; j < z; j++) {
        data(i, j) = 7;
      }
    }
  }

  // Setup the rhs tensor of data
  plaidml::tensor<float> c = dev_.allocate(plaidml::shape<float>(ctx, {z, y}));
  {
    plaidml::mapping<float> data = c.map(plaidml::map_for_write);
    for (size_t i = 0; i < z; i++) {
      for (size_t j = 0; j < y; j++) {
        data(i, j) = 7;
      }
    }
  }

  plaidml::tensor<float> output = dev_.allocate(plaidml::shape<float>(ctx, {x, y}));

  plaidml::invoker(ctx, matmul_).set_input("B", b).set_input("C", c).set_output("A", output).invoke();

  {
    plaidml::mapping<float> data = output.map(plaidml::map_for_read);
    for (size_t i = 0; i < x; i++) {
      for (size_t j = 0; j < y; j++) {
        EXPECT_FLOAT_EQ(data(i, j), 7 * 7 * z);
      }
    }
  }
}

INSTANTIATE_TEST_CASE_P(MatrixSize, MatMulTest,
                        ::testing::Combine(::testing::Range(static_cast<size_t>(1), static_cast<size_t>(1024), 61),
                                           ::testing::Range(static_cast<size_t>(1), static_cast<size_t>(1024), 79),
                                           ::testing::Range(static_cast<size_t>(1), static_cast<size_t>(1024), 57)));

}  // namespace
