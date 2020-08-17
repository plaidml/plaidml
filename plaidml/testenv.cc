// Copyright 2019, Intel Corp.

#include <gflags/gflags.h>
#include <gmock/gmock.h>

#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"

DEFINE_string(plaidml_device, "llvm_cpu.0", "The device to run tests on");
DEFINE_string(plaidml_target, "llvm_cpu", "The compilation target");

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
