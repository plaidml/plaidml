// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "tile/platform/local_machine/scheduler.h"

namespace vertexai {
namespace tile {
namespace local_machine {

typedef std::tuple<std::shared_ptr<Scheduler>, tile::proto::Program> SchedulerTestParam;

void PrintTo(const SchedulerTestParam& param, ::std::ostream* os);

// Scheduler implementation conformance tests.
//
// To test a scheduler, #include this header (linking with the
// :scheduler_tests target), and use INSTANTIATE_TEST_CASE_P to
// instantiate the conformance tests with a Scheduler instance -- e.g.
//
//   INSTANTIATE_TEST_CASE_P(
//       MyScheduler,
//       SchedulerTest,
//       ::testing::Combine(
//         ::testing::Values(std::make_shared<MyScheduler>()),
//         ::testing::ValuesIn(SchedulerTest::GetTestPrograms())));
//
class SchedulerTest : public ::testing::TestWithParam<SchedulerTestParam> {
 public:
  static std::vector<tile::proto::Program> GetTestPrograms();

 protected:
  std::shared_ptr<Scheduler> GetScheduler() { return std::get<0>(GetParam()); }
  const tile::proto::Program& GetProgram() { return std::get<1>(GetParam()); }
  tile::lang::HardwareSettings GetSettings();
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
