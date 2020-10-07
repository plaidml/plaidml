#pragma once

#include <gflags/gflags.h>
#include <gmock/gmock.h>

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "plaidml/exec/exec.h"

DECLARE_bool(generate_filecheck_input);

namespace plaidml::edsl {

using MultiBuffer = std::variant<  //
    std::vector<float>,            //
    std::vector<double>,           //
    std::vector<int8_t>,           //
    std::vector<int16_t>,          //
    std::vector<int32_t>,          //
    std::vector<int64_t>,          //
    std::vector<uint8_t>,          //
    std::vector<uint16_t>,         //
    std::vector<uint32_t>,         //
    std::vector<uint64_t>>;

using TensorBuffers = std::vector<MultiBuffer>;

class TestFixture : public ::testing::Test {
 protected:
  template <typename T>
  void compareElements(T a, T b) {
    EXPECT_EQ(a, b);
  }

  void compareElements(float a, float b) {
    if (isfinite(a) && isfinite(b)) {
      EXPECT_NEAR(a, b, (fabs(a) + fabs(b)) / 10000.0);
    } else {
      EXPECT_EQ(a, b);
    }
  }

  void compareElements(double a, double b) {
    if (isfinite(a) && isfinite(b)) {
      EXPECT_NEAR(a, b, (fabs(a) + fabs(b)) / 10000.0);
    } else {
      EXPECT_EQ(a, b);
    }
  }

  template <typename T>
  void compareBuffers(plaidml::Buffer buffer, const std::vector<T>& expected) {
    ASSERT_EQ(buffer.size(), expected.size() * sizeof(expected[0]));
    auto data = reinterpret_cast<T*>(buffer.data());
    std::vector<T> actual(data, data + expected.size());
    for (size_t i = 0; i < actual.size(); i++) {
      compareElements(actual[i], expected[i]);
    }
  }

  void checkProgram(Program program, const TensorBuffers& inputs, const TensorBuffers& expected) {
    program.compile();
#if !defined(_WIN32)
    std::vector<Buffer> input_buffers;
    auto input_shapes = program.inputs();
    ASSERT_EQ(inputs.size(), input_shapes.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      std::visit(
          [&](auto&& vec) {
            Buffer buffer{vec, input_shapes[i]};
            input_buffers.emplace_back(buffer);
          },
          inputs[i]);
    }

    std::vector<Buffer> output_buffers;
    for (auto shape : program.outputs()) {
      output_buffers.emplace_back(shape);
    }
    auto executable = exec::Executable(program, input_buffers, output_buffers);
    executable.run();

    ASSERT_EQ(expected.size(), program.outputs().size());
    for (size_t i = 0; i < expected.size(); i++) {
      std::visit([&](auto&& vec) { compareBuffers(output_buffers[i], vec); }, expected[i]);
    }
#endif
  }

  Program makeProgram(const std::string& name, const std::vector<Tensor>& inputs, const std::vector<Tensor>& outputs) {
    auto program = edsl::buildProgram(name, inputs, outputs);
    writeForFileCheck(program);
    return program;
  }

  void writeForFileCheck(const Program& program) {
    if (FLAGS_generate_filecheck_input) {
      std::cout << program << std::endl;
    }
  }

  void runProgram(Program program) {
    program.compile();
#if !defined(_WIN32)
    std::vector<Buffer> inputs;
    for (const TensorShape& shape : program.inputs()) {
      inputs.emplace_back(shape);
    }
    std::vector<Buffer> outputs;
    for (const TensorShape& shape : program.outputs()) {
      outputs.emplace_back(shape);
    }
    exec::Executable(program, inputs, outputs).run();
#endif
  }
};

}  // namespace plaidml::edsl
