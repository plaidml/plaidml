// Copyright 2020, Intel Corp.

#include <gflags/gflags.h>
#include <gmock/gmock.h>

// TODO: The awkward include of a *.cpp is to get `ngraph_register_interpreter_backend`. We could instead duplicate the
// function's code to only include headers, but I don't think that's actually cleaner.
#include "ngraph/test/runtime/interpreter/int_backend.cpp"  // NOLINT[build/include]

namespace {

class Environment : public ::testing::Environment {
  void SetUp() override { ngraph_register_interpreter_backend(); }
};

[[gnu::unused]] auto init = []() {
  ::testing::AddGlobalTestEnvironment(new Environment);
  return 0;
}();

}  // namespace
