#pragma once

#include <gmock/gmock.h>

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "plaidml/exec/exec.h"
#include "pmlc/util/logging.h"

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

using TensorBuffers = std::map<TensorRef, MultiBuffer>;

class TestFixture : public ::testing::Test {
 protected:
  template <typename T>
  void compareElements(T a, T b) {
    EXPECT_EQ(a, b);
  }

  void compareElements(float a, float b) { EXPECT_NEAR(a, b, (fabs(a) + fabs(b)) / 10000.0); }
  void compareElements(double a, double b) { EXPECT_NEAR(a, b, (fabs(a) + fabs(b)) / 10000.0); }

  template <typename T>
  void compareBuffers(plaidml::View view, const std::vector<T>& expected) {
    ASSERT_THAT(view.size(), expected.size() * sizeof(expected[0]));
    auto data = reinterpret_cast<T*>(view.data());
    std::vector<T> actual(data, data + expected.size());
    IVLOG(3, "Expected: " << expected);
    IVLOG(3, "Actual  : " << actual);
    for (size_t i = 0; i < actual.size(); i++) {
      compareElements(actual[i], expected[i]);
    }
  }

  void checkProgram(                //
      const Program& program,       //
      const TensorBuffers& inputs,  //
      const TensorBuffers& expected) {
#if !defined(_WIN32)
    auto binder = exec::Binder(program);
    auto executable = binder.compile();
    for (const auto& kvp : inputs) {
      std::visit([&](auto&& vec) { binder.input(kvp.first).copy_from(vec.data()); }, kvp.second);
    }
    executable->run();
    for (auto kvp : expected) {
      auto view = binder.output(kvp.first).mmap_current();
      std::visit([&](auto&& vec) { compareBuffers(view, vec); }, kvp.second);
    }
#endif
  }

  Program makeProgram(const std::string& name, const std::vector<Tensor>& outputs) {
    ProgramBuilder builder(name, outputs);
    shimTarget(builder);
    auto program = builder.compile();
    std::cout << program << std::endl;
    return program;
  }

  void runProgram(const Program& program) {
#if !defined(_WIN32)
    exec::Binder(program).compile()->run();
#endif
  }

  // Sets a skip if the current target is the specified target,
  // returning true iff the current test should be skipped.
  //
  // N.B. The test itself continues to execute, and should use
  //      IsSkipped() or the result of this method to exit when
  //      necessary.  This is useful for tests whose output will be
  //      processed by LLVM FileCheck.
  //
  // For example, if the test's output is being checked against its
  // source by FileCheck, one might want to use:
  //
  //     setIsSkipped("some_target");
  //
  //     // Code that emits some program, often via makeProgram()
  //
  //     // CHECK-LABEL: MyTest
  //
  //     if (IsSkipped()) {
  //       return;
  //     }
  //
  //     // Code to run the program and check its output.
  //
  // On the other hand, if the test's output isn't being checked
  // against its source by FileCheck, one can simply use:
  //
  //    if (setIsSkipped("some_target")) {
  //      return;
  //    }
  //
  bool setSkipOnTarget(const char* target) {
    if (Settings::get("PLAIDML_TARGET") == target) {
      [target]() -> void {
        GTEST_SKIP() << "Test skipped on " << target;
      }();
      skipTarget_ = target;
      return true;
    }
    return false;
  }

  // Sets the target to llvm_cpu if the current actual PlaidML target is being skipped.
  void shimTarget(ProgramBuilder& pb) {
    if (skipTarget_.size()) {
      pb.target("llvm_cpu");
    }
  }

 private:
  std::string skipTarget_;
};

}  // namespace plaidml::edsl
