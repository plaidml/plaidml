// Copyright 2018, Intel Corporation.
#include <gmock/gmock.h>

#include "tile/platform/local_machine/fifo_scheduler.h"
#include "tile/platform/local_machine/scheduler_test.h"

using ::testing::AnyOf;
using ::testing::Combine;
using ::testing::Eq;
using ::testing::UnorderedElementsAre;
using ::testing::Values;
using ::testing::ValuesIn;

namespace vertexai {
namespace tile {
namespace local_machine {
namespace fifo_scheduler {
namespace {

hal::proto::HardwareSettings TestHardwareSettings() {
  hal::proto::HardwareSettings result;
  result.set_goal_groups(32);
  return result;
}

INSTANTIATE_TEST_CASE_P(FifoScheduler, SchedulerTest,
                        Combine(Values(std::make_shared<FifoScheduler>(std::kilo::num, std::giga::num,
                                                                       TestHardwareSettings())),
                                ValuesIn(SchedulerTest::GetTestPrograms())));

class InitStepTest : public ::testing::Test {
 protected:
  schedule::Alloc* AddTmp(std::uint64_t size) {
    auto it = schedule_.allocs.emplace(schedule_.allocs.end(), schedule::Alloc{});
    it->byte_size = size;
    return &*it;
  }

  schedule::Alloc* AddProgramInput(std::uint64_t size, const char* name) {
    auto alloc = AddTmp(size);
    alloc->input = name;
    return alloc;
  }

  void DoInitSteps() { InitSteps(&b_, schedule_); }

  tile::proto::Program program_;
  lang::KernelList kl_;
  Build b_{program_, kl_, 0, std::kilo::num, std::giga::num, 16};
  schedule::Schedule schedule_;
};

TEST_F(InitStepTest, ReadyStep) {
  auto step = schedule_.steps.emplace(schedule_.steps.end(), schedule::Step::Tag::kRun);
  step->inputs.push_back(AddProgramInput(1024, "X_I0"));
  step->outputs.push_back(schedule::OutputInfo{AddTmp(1024), false});
  DoInitSteps();
  ASSERT_THAT(b_.pending_steps_storage.size(), Eq(1));
  auto psit = b_.pending_steps_storage.begin();
  EXPECT_THAT(psit->dependency_count, Eq(0));
  EXPECT_THAT(psit->step, Eq(&*step));
  ASSERT_THAT(b_.locs.size(), Eq(2));
  auto lit = b_.locs.begin();
  EXPECT_THAT(lit->byte_size, Eq(1024));
  ++lit;
  EXPECT_THAT(lit->byte_size, Eq(1024));
}

TEST_F(InitStepTest, AliasStep) {
  auto step = schedule_.steps.emplace(schedule_.steps.end(), schedule::Step::Tag::kRun);
  auto* input = AddProgramInput(1024, "X_I0");
  step->inputs.push_back(input);
  auto* output = AddTmp(1024);
  step->outputs.push_back(schedule::OutputInfo{output, false});
  output->safe_self_alias_allocs.insert(input);
  DoInitSteps();
  ASSERT_THAT(b_.pending_steps_storage.size(), Eq(1));
  auto psit = b_.pending_steps_storage.begin();
  EXPECT_THAT(psit->dependency_count, Eq(0));
  EXPECT_THAT(psit->step, Eq(&*step));
  ASSERT_THAT(b_.locs.size(), Eq(1));
  EXPECT_THAT(b_.locs.front().byte_size, Eq(1024));
}

TEST_F(InitStepTest, ZeroStep) {
  auto s1 = schedule_.steps.emplace(schedule_.steps.end(), schedule::Step::Tag::kRun);
  auto* tmp = AddTmp(1024);
  s1->outputs.push_back(schedule::OutputInfo{tmp, false});
  auto s2 = schedule_.steps.emplace(schedule_.steps.end(), schedule::Step::Tag::kRun);
  s2->inputs.push_back(tmp);
  auto* input = AddProgramInput(1024, "X_I0");
  s2->inputs.push_back(input);
  tmp = AddTmp(1024);
  s2->outputs.push_back(schedule::OutputInfo{tmp, false});
  DoInitSteps();
  ASSERT_THAT(b_.pending_steps_storage.size(), Eq(2));
  auto psit = b_.pending_steps_storage.begin();
  EXPECT_THAT(psit->dependency_count, Eq(0));
  EXPECT_THAT(psit->step, Eq(&*s1));
  ++psit;
  EXPECT_THAT(psit->dependency_count, Eq(1));
  EXPECT_THAT(psit->step, Eq(&*s2));
  ASSERT_THAT(b_.locs.size(), Eq(3));
}

TEST_F(InitStepTest, ReuseAllocs) {
  auto s1 = schedule_.steps.emplace(schedule_.steps.end(), schedule::Step::Tag::kRun);
  auto* input = AddProgramInput(1024, "X_I0");
  s1->inputs.push_back(input);
  auto* tmp = AddTmp(1024);
  s1->outputs.push_back(schedule::OutputInfo{tmp, false});
  auto s2 = schedule_.steps.emplace(schedule_.steps.end(), schedule::Step::Tag::kRun);
  s2->inputs.push_back(tmp);
  auto* output = AddTmp(1024);
  s2->outputs.push_back(schedule::OutputInfo{output, false});
  DoInitSteps();
  ASSERT_THAT(b_.pending_steps_storage.size(), Eq(2));
  auto psit = b_.pending_steps_storage.begin();
  EXPECT_THAT(psit->dependency_count, Eq(0));
  EXPECT_THAT(psit->step, Eq(&*s1));
  ++psit;
  EXPECT_THAT(psit->dependency_count, Eq(1));
  EXPECT_THAT(psit->step, Eq(&*s2));
  ASSERT_THAT(b_.locs.size(), Eq(2));
  auto lit = b_.locs.begin();
  EXPECT_THAT(lit->byte_size, Eq(1024));
  ++lit;
  EXPECT_THAT(lit->byte_size, Eq(1024));
}

class RunnableStepsTest : public ::testing::Test {
 protected:
  std::vector<PendingStep*> BuildRunnable(std::list<PendingStep>* steps) {
    auto pending = InitPendingSteps(steps);
    RunnableSteps runnable{&pending};
    return std::vector<PendingStep*>(runnable.begin(), runnable.end());
  }
};

TEST_F(RunnableStepsTest, NoSteps) {
  std::list<PendingStep> steps{};
  auto runnable = BuildRunnable(&steps);
  EXPECT_THAT(runnable.size(), Eq(0));
}

TEST_F(RunnableStepsTest, ManyRunnable) {
  std::list<PendingStep> steps{
      PendingStep{1, nullptr, 0, false},  PendingStep{2, nullptr, 0, false}, PendingStep{3, nullptr, 0, false},
      PendingStep{4, nullptr, 0, false},  PendingStep{5, nullptr, 0, false}, PendingStep{6, nullptr, 0, false},
      PendingStep{7, nullptr, 0, false},  PendingStep{8, nullptr, 0, false}, PendingStep{9, nullptr, 0, false},
      PendingStep{10, nullptr, 0, false},
  };
  auto runnable = BuildRunnable(&steps);
  EXPECT_THAT(runnable.size(), Eq(10));
}

TEST_F(RunnableStepsTest, SomeRunnable) {
  std::list<PendingStep> steps{
      PendingStep{1, nullptr, 0, false}, PendingStep{0, nullptr, 1, false}, PendingStep{2, nullptr, 0, false},
      PendingStep{0, nullptr, 3, false}, PendingStep{0, nullptr, 1, false}, PendingStep{0, nullptr, 3, false},
      PendingStep{0, nullptr, 2, false}, PendingStep{3, nullptr, 0, false}, PendingStep{0, nullptr, 9, false},
      PendingStep{4, nullptr, 0, false},
  };
  auto runnable = BuildRunnable(&steps);
  ASSERT_THAT(runnable.size(), Eq(4));
  auto sit = steps.begin();
  auto s0 = &*sit;
  std::advance(sit, 2);
  auto s2 = &*sit;
  std::advance(sit, 5);
  auto s7 = &*sit;
  std::advance(sit, 2);
  auto s9 = &*sit;
  EXPECT_THAT(runnable, UnorderedElementsAre(s0, s2, s7, s9));
}

}  // namespace
}  // namespace fifo_scheduler
}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
