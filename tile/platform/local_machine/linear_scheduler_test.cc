// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/linear_scheduler.h"

#include <ratio>

#include "tile/platform/local_machine/block_placer.h"
#include "tile/platform/local_machine/naive_placer.h"
#include "tile/platform/local_machine/scheduler_test.h"

using ::testing::Combine;
using ::testing::Values;
using ::testing::ValuesIn;

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

INSTANTIATE_TEST_CASE_P(
    LinearScheduler, SchedulerTest,
    Combine(Values(std::make_shared<LinearScheduler>(std::make_shared<NaivePlacer>(std::kilo::num)),
                   std::make_shared<LinearScheduler>(std::make_shared<BlockPlacer>(std::kilo::num))),
            ValuesIn(SchedulerTest::GetTestPrograms())));

}  // namespace
}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
