#pragma once

#include <gflags/gflags.h>
#include <gmock/gmock.h>

#include <string>
#include <variant>
#include <vector>

#include "half.hpp"
#include "plaidml/edsl/edsl.h"

DECLARE_bool(generate_filecheck_input);

namespace plaidml {

using MultiBuffer = std::variant<   //
    std::vector<half_float::half>,  //
    std::vector<float>,             //
    std::vector<double>,            //
    std::vector<int8_t>,            //
    std::vector<int16_t>,           //
    std::vector<int32_t>,           //
    std::vector<int64_t>,           //
    std::vector<uint8_t>,           //
    std::vector<uint16_t>,          //
    std::vector<uint32_t>,          //
    std::vector<uint64_t>>;

using TensorBuffers = std::vector<MultiBuffer>;

class TestFixture : public ::testing::Test {
 protected:
  void checkExact(Program program, const TensorBuffers& inputs, const TensorBuffers& expected);

  void checkClose(Program program, const TensorBuffers& inputs, const TensorBuffers& expected, double tolerance = 1e-5);

  Program makeProgram(const std::string& name, const std::vector<edsl::Tensor>& inputs,
                      const std::vector<edsl::Tensor>& outputs);

  void writeForFileCheck(const Program& program);

  void runProgram(Program program);
};

}  // namespace plaidml
