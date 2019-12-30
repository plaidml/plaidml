// Copyright 2019, Intel Corp.

#include <gmock/gmock.h>

#include "plaidml2/edsl/edsl.h"
#include "plaidml2/exec/exec.h"
#include "plaidml2/op/op.h"

namespace {

class Environment : public ::testing::Environment {
  void SetUp() override {
    plaidml2::init();
    plaidml2::edsl::init();
    plaidml2::op::init();
    plaidml2::exec::init();
  }
};

[[gnu::unused]] auto init = []() {
  ::testing::AddGlobalTestEnvironment(new Environment);
  return 0;
}();

}  // namespace
