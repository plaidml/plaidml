// Copyright 2020 Intel Corporation
//
// N.B. When running via lit, we always use the llvm_cpu device.
// RUN: cc_test \
// RUN:   --plaidml_device=llvm_cpu.0 \
// RUN:   --plaidml_target=llvm_cpu \
// RUN:   --generate_filecheck_input \
// RUN: | FileCheck %s

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"
#include "plaidml/testenv.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

using ::testing::HasSubstr;

#if ERRORTRACING
#define EXPECT_ERROR_LINE(errmsg, eline) EXPECT_THAT(errmsg, HasSubstr(std::to_string(eline)))
#else
#define EXPECT_ERROR_LINE(errmsg, eline) EXPECT_THAT(errmsg, HasSubstr(":0"));
#endif

namespace plaidml::edsl {

namespace {

class CppEdsl : public TestFixture {};

TEST_F(CppEdsl, BindDims) {
  const char* errmsg;
  int eline;
  try {
    auto X = Placeholder(DType::FLOAT32, {10, 10});
    auto Y = Placeholder(DType::FLOAT32, {12, 10});
    TensorDim I, J, K;
    TensorIndex i, j, k;
    X.bind_dims({I, K});
    // clang-format off
    eline = __LINE__; Y.bind_dims({K, J});
    // clang-format on
  } catch (const std::exception& e) {
    errmsg = e.what();
  }
  EXPECT_ERROR_LINE(errmsg, eline);
}

TEST_F(CppEdsl, EltwiseMismatch) {
  const char* errmsg;
  int eline;
  try {
    auto X = Placeholder(DType::FLOAT32, {10, 10});
    auto Y = Placeholder(DType::FLOAT32, {12, 10});
    // clang-format off
    eline = __LINE__; auto O = X + Y;
    // clang-format on
  } catch (const std::exception& e) {
    errmsg = e.what();
  }
  EXPECT_ERROR_LINE(errmsg, HasSubstr(std::to_string(eline)));
}

TEST_F(CppEdsl, OpOperators) {
  const char* errmsg;
  int eline;
  try {
    auto X = Placeholder(DType::FLOAT32, {10, 10});
    auto Y = Placeholder(DType::FLOAT32, {12, 12});
    auto RX = plaidml::op::relu(X);
    auto RY = plaidml::op::relu(Y);
    // clang-format off
    eline = __LINE__; auto O = X + Y;
    // clang-format on
  } catch (const std::exception& e) {
    errmsg = e.what();
  }
  EXPECT_ERROR_LINE(errmsg, HasSubstr(std::to_string(eline)));
}

}  // namespace
}  // namespace plaidml::edsl
