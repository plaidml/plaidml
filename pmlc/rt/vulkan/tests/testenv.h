#pragma once

#include <gmock/gmock.h>

#include <map>
#include <variant>
#include <vector>

#include "plaidml/exec/exec.h"

namespace plaidml::edsl {

using MultiBuffer = std::variant< //
    std::vector<float>,           //
    std::vector<double>,          //
    std::vector<int8_t>,          //
    std::vector<int16_t>,         //
    std::vector<int32_t>,         //
    std::vector<int64_t>,         //
    std::vector<uint8_t>,         //
    std::vector<uint16_t>,        //
    std::vector<uint32_t>,        //
    std::vector<uint64_t>>;

using TensorBuffers = std::map<TensorRef, MultiBuffer>;

class TestFixture : public ::testing::Test {
protected:
  template <typename T> void compareElements(T a, T b) { EXPECT_EQ(a, b); }

  void compareElements(float a, float b) {
    EXPECT_NEAR(a, b, (fabs(a) + fabs(b)) / 10000.0);
  }
  void compareElements(double a, double b) {
    EXPECT_NEAR(a, b, (fabs(a) + fabs(b)) / 10000.0);
  }

  template <typename T>
  void compareBuffers(plaidml::View view, const std::vector<T> &expected) {
    ASSERT_THAT(view.size(), expected.size() * sizeof(expected[0]));
    auto data = reinterpret_cast<T *>(view.data());
    std::vector<T> actual(data, data + expected.size());
    for (size_t i = 0; i < actual.size(); i++) {
      compareElements(actual[i], expected[i]);
    }
  }

  void checkProgram(               //
      const Program &program,      //
      const TensorBuffers &inputs, //
      const TensorBuffers &expected) {
#if !defined(_WIN32)
    auto binder = exec::Binder(program);
    auto executable = binder.compile();
    for (const auto &kvp : inputs) {
      std::visit(
          [&](auto &&vec) { binder.input(kvp.first).copy_from(vec.data()); },
          kvp.second);
    }
    executable->run();
    for (auto kvp : expected) {
      auto view = binder.output(kvp.first).mmap_current();
      std::visit([&](auto &&vec) { compareBuffers(view, vec); }, kvp.second);
    }
#endif
  }

  void runProgram(const Program &program) {
#if !defined(_WIN32)
    exec::Binder(program).compile()->run();
#endif
  }
};

} // namespace plaidml::edsl
