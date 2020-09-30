// Copyright 2020, Intel Corp.

#include <gflags/gflags.h>
#include <gmock/gmock.h>

#include "ngraph/test/runtime/interpreter/int_backend.hpp"

namespace {

class Environment : public ::testing::Environment {
  void SetUp() override {
    // TODO: This is a duplication of `ngraph_register_interpreter_backend` from
    // "ngraph/test/runtime/interpreter/int_backend.cpp" because it's not available in headers
    ngraph::runtime::BackendManager::register_backend("INTERPRETER", [](const std::string& /* config */) {
      return std::make_shared<ngraph::runtime::interpreter::INTBackend>();
    });
  }
};

[[gnu::unused]] auto init = []() {
  ::testing::AddGlobalTestEnvironment(new Environment);
  return 0;
}();

}  // namespace
