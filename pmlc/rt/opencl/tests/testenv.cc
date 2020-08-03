// Copyright 2019, Intel Corp.

#include <gmock/gmock.h>

#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"

#include "pmlc/rt/opencl/tests/testenv.h"

namespace {

class Environment : public ::testing::Environment {
  void SetUp() override {
    plaidml::init();
    plaidml::edsl::init();
    plaidml::op::init();
    plaidml::exec::init();
    plaidml::Settings::set("PLAIDML_DEVICE", "intel_gen_ocl_spirv");
    plaidml::Settings::set("PLAIDML_TARGET", "intel_gen_ocl_spirv");
  }
};

[[gnu::unused]] auto init = []() {
  ::testing::AddGlobalTestEnvironment(new Environment);
  return 0;
}();

} // namespace
