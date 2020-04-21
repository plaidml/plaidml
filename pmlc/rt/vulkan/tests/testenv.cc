// Copyright 2019, Intel Corp.

#include <gmock/gmock.h>

#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"

#include "pmlc/rt/vulkan/tests/testenv.h"

namespace {

class Environment : public ::testing::Environment {
  void SetUp() override {
    plaidml::init();
    plaidml::edsl::init();
    plaidml::op::init();
    plaidml::exec::init();
    plaidml::Settings::set("PLAIDML_DEVICE", "intel_gen");
    plaidml::Settings::set("PLAIDML_TARGET", "intel_gen");
  }
};

[[gnu::unused]] auto init = []() {
  ::testing::AddGlobalTestEnvironment(new Environment);
  return 0;
}();

} // namespace
