// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/loose_scheduler.h"

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
    LooseScheduler, SchedulerTest,
    Combine(Values(std::make_shared<LooseScheduler>(std::make_shared<NaivePlacer>(std::kilo::num), 4 * std::giga::num),
                   std::make_shared<LooseScheduler>(std::make_shared<BlockPlacer>(std::kilo::num), 4 * std::giga::num)),
            ValuesIn(SchedulerTest::GetTestPrograms())));

}  // namespace
}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
