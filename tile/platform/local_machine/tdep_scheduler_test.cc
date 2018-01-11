// Copyright 2017, Vertex.AI.

#include "tile/platform/local_machine/tdep_scheduler.h"

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
    TransitiveDepScheduler, SchedulerTest,
    Combine(Values(std::make_shared<TransitiveDepScheduler>(std::make_shared<NaivePlacer>(std::kilo::num), 16),
                   std::make_shared<TransitiveDepScheduler>(std::make_shared<BlockPlacer>(std::kilo::num), 16)),
            ValuesIn(SchedulerTest::GetTestPrograms())));

}  // namespace
}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
