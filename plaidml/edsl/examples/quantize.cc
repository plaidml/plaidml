// Copyright 2020 Intel Corporation.
// Note:
//    This file is being used by sphinx docs to pull in code blocks.
//    Code blocks are pulled into docs/usage/*.rst
//    Any changes made here may upset the docs.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "plaidml/edsl/edsl.h"
#include "plaidml/testenv.h"

namespace plaidml::edsl {

// quantize_float32_int8_start
Tensor QuantizeFloat32ToInt8(const Tensor& A, float range) {
  Tensor O = A / range * 128;
  Tensor O_int = edsl::cast(O, DType::INT8);
  return O_int;
}
// quantize_float32_int8_end

class ExampleCppEdsl : public TestFixture {};

TEST_F(ExampleCppEdsl, Quantize) {
  auto A = Placeholder(DType::FLOAT32, {3});
  auto program = makeProgram("quantize", {A}, {QuantizeFloat32ToInt8(A, 0.5)});

  std::vector<float> data_A{0.1, 0.4, 0.3};
  std::vector<int8_t> data_O{25, 102, 76};
  checkExact(program, {data_A}, {data_O});
}

}  // namespace plaidml::edsl
