// Copyright 2019, Intel Corp.

#include "plaidml/testenv.h"

#include "llvm/ADT/STLExtras.h"

#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"

DEFINE_string(plaidml_device, "llvm_cpu.0", "The device to run tests on");
DEFINE_string(plaidml_target, "llvm_cpu", "The compilation target");
DEFINE_bool(generate_filecheck_input, false, "Write test programs as MLIR for FileCheck");

namespace {

class Environment : public ::testing::Environment {
  void SetUp() override {
    plaidml::init();
    plaidml::edsl::init();
    plaidml::op::init();
    plaidml::exec::init();
    plaidml::Settings::set("PLAIDML_DEVICE", FLAGS_plaidml_device);
    plaidml::Settings::set("PLAIDML_TARGET", FLAGS_plaidml_target);
  }
};

[[gnu::unused]] auto init = []() {
  ::testing::AddGlobalTestEnvironment(new Environment);
  return 0;
}();

}  // namespace

namespace std {

ostream& operator<<(ostream& os, half_float::half value) {
  os << static_cast<double>(value);
  return os;
}

}  // namespace std

namespace plaidml {

template <typename T>
static void compareExact(plaidml::Buffer buffer, const std::vector<T>& expected) {
  ASSERT_EQ(buffer.size(), expected.size() * sizeof(expected[0]));
  auto data = reinterpret_cast<T*>(buffer.data());
  std::vector<T> actual(data, data + expected.size());
  EXPECT_THAT(actual, ::testing::ContainerEq(expected));
}

template <typename T>
static void compareClose(plaidml::Buffer buffer, const std::vector<T>& expected, double tolerance) {
  ASSERT_EQ(buffer.size(), expected.size() * sizeof(expected[0]));
  auto data = reinterpret_cast<T*>(buffer.data());
  std::vector<T> actual(data, data + expected.size());
  ASSERT_EQ(actual.size(), expected.size());
  for (auto [x, y] : llvm::zip(actual, expected)) {  // NOLINT[whitespace/braces]
    if (isfinite(x) && isfinite(y)) {
      EXPECT_NEAR(x, y, (fabs(x) + fabs(y)) * tolerance);
    } else {
      EXPECT_EQ(x, y);
    }
  }
}

struct Runner {
  explicit Runner(Program program) : program(program) { program.compile(); }

  void run(const TensorBuffers& inputs) {
#if !defined(_WIN32)
    std::vector<Buffer> inputBuffers;
    auto inputShapes = program.inputs();
    ASSERT_EQ(inputs.size(), inputShapes.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      std::visit(
          [&](auto&& vec) {
            Buffer buffer{vec, inputShapes[i]};
            inputBuffers.emplace_back(buffer);
          },
          inputs[i]);
    }

    for (auto shape : program.outputs()) {
      outputBuffers.emplace_back(shape);
    }
    auto executable = exec::Executable(program);
    executable.run(inputBuffers, outputBuffers);
#endif
  }

  void checkExact(const TensorBuffers& expected) {
    ASSERT_EQ(expected.size(), program.outputs().size());
#if !defined(_WIN32)
    for (size_t i = 0; i < expected.size(); i++) {
      std::visit([&](auto&& vec) { compareExact(outputBuffers[i], vec); }, expected[i]);
    }
#endif
  }

  void checkClose(const TensorBuffers& expected, double tolerance) {
    ASSERT_EQ(expected.size(), program.outputs().size());
#if !defined(_WIN32)
    for (size_t i = 0; i < expected.size(); i++) {
      std::visit([&](auto&& vec) { compareClose(outputBuffers[i], vec, tolerance); }, expected[i]);
    }
#endif
  }

  Program program;
  std::vector<Buffer> outputBuffers;
};

void TestFixture::checkExact(Program program, const TensorBuffers& inputs, const TensorBuffers& expected) {
  Runner runner(program);
  runner.run(inputs);
  runner.checkExact(expected);
}

void TestFixture::checkClose(Program program, const TensorBuffers& inputs, const TensorBuffers& expected,
                             double tolerance) {
  Runner runner(program);
  runner.run(inputs);
  runner.checkClose(expected, tolerance);
}

Program TestFixture::makeProgram(const std::string& name, const std::vector<edsl::Tensor>& inputs,
                                 const std::vector<edsl::Tensor>& outputs) {
  auto program = edsl::buildProgram(name, inputs, outputs);
  writeForFileCheck(program);
  return program;
}

void TestFixture::writeForFileCheck(const Program& program) {
  if (FLAGS_generate_filecheck_input) {
    std::cout << program << std::endl;
  }
}

void TestFixture::runProgram(Program program) {
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
  exec::Executable(program).run(inputs, outputs);
#endif
}

}  // namespace plaidml
