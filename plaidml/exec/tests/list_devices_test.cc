// Copyright 2020 Intel Corporation

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "plaidml/exec/exec.h"

using ::testing::Contains;

namespace plaidml::exec {
namespace {

TEST(ListDevices, DevicesExist) {
  auto devices = list_devices();
  EXPECT_THAT(devices, Contains("llvm_cpu.0"));
}

class Environment : public ::testing::Environment {
  void SetUp() override { plaidml::exec::init(); }
};

[[gnu::unused]] auto init = []() {
  ::testing::AddGlobalTestEnvironment(new Environment);
  return 0;
}();

}  // namespace
}  // namespace plaidml::exec
