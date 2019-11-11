// Copyright 2019, Intel Corp.

#include <gmock/gmock.h>

#include "plaidml2/edsl/edsl.h"
#include "plaidml2/op/op.h"

namespace {

class Environment : public ::testing::Environment {
  void SetUp() override {
    plaidml::init();
    plaidml::edsl::init();
    plaidml::op::init();
  }
};

[[gnu::unused]] auto init = []() {
  ::testing::AddGlobalTestEnvironment(new Environment);
  return 0;
}();

}  // namespace
