// Copyright 2018, Intel Corporation.

#pragma once

#include <list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tile/platform/local_machine/scheduler.h"

namespace vertexai {
namespace tile {
namespace local_machine {
namespace fifo_scheduler {

// Tracks the state of an alloc-to-be-created.
struct Loc {
  explicit Loc(std::uint64_t byte_size_, bool is_io_ = false) : byte_size{byte_size_}, is_io{is_io_} {}
  std::uint64_t byte_size;
  bool is_io;
  bool is_host = false;
  bool add_dep = false;
  schedule::Alloc* contents = nullptr;
  std::string input;
  std::string output;
  std::uint64_t cache_memtime = 0;
};

// Tracks the state of a step that hasn't been scheduled yet (but needs to be).
struct PendingStep {
  std::size_t loc;
  const schedule::Step* step;

  // #Inputs this step is waiting on.
  std::size_t dependency_count;

  // True iff this is a zero kernel.
  bool is_zero;

  // Zero-input generators for this step.
  std::list<std::pair<PendingStep*, schedule::Alloc*>> zero_inputs;

  // Steps waiting on this step's outputs.  Note that a dependent will be listed multiple times if
  // it is waiting on multiple outputs from this step.
  std::list<PendingStep*> dependents;

  // The maximum distance from this step to all of its downstream outputs.
  std::uint64_t distance;

  // Computed work groups (max of step and device work groups)
  std::uint64_t work_groups;
};

// Tracks the state of a step that's been scheduled.
struct ScheduledStep {
  const schedule::Step* step;
  std::uint64_t work_groups;
  std::list<PendingStep*> dependents;
  std::vector<Loc*> outputs;
};

std::vector<PendingStep*> InitPendingSteps(std::list<PendingStep>* pending_steps_storage);

// Removes a PendingStep from a pending step heap.  This is used when a step moves from
// the runnable state to the running state.
void RemovePendingStep(std::vector<PendingStep*>* pending, PendingStep* ps);

// Resolves one pending dependency for the indicated step, possibly making it runnable.
void ResolveDep(std::vector<PendingStep*>* pending, PendingStep* ps);

// Computes the new dependencies required by the schedule.
void AddDeps(schedule::Schedule* schedule);

// Represents a schedule build in progress; provides mid-level manipulators to help translate
// high-level actions to low-level datastructure operations.
struct Build {
  Build(const tile::proto::Program& program_, const lang::KernelList& kl_, std::size_t kernel_count_,
        std::size_t alignment_, std::uint64_t mem_available_, std::uint64_t work_group_limit_)
      : program{&program_},
        kl{&kl_},
        running{scheduled.end()},
        alignment{alignment_},
        mem_available{mem_available_},
        work_group_limit{work_group_limit_} {}

  const tile::proto::Program* program;
  const lang::KernelList* kl;
  std::list<PendingStep> pending_steps_storage;
  std::vector<PendingStep*> pending;
  std::list<ScheduledStep> scheduled;
  std::unordered_map<schedule::Alloc*, Loc*> value_locs;
  std::multimap<std::uint64_t, Loc*> free_locs;
  std::unordered_map<schedule::Alloc*, Loc*> input_locs;
  std::list<Loc> locs;
  std::unordered_map<schedule::Alloc*, std::size_t> alloc_refcounts;
  std::list<ScheduledStep>::iterator running;
  std::uint64_t running_groups = 0;
  std::size_t alignment;
  std::uint64_t mem_available;
  std::uint64_t work_group_limit;
  std::uint64_t current_memtime = 0;
};

void InitPendingSteps(Build* b);

void InitStep(Build* b, const schedule::Step& step,
              std::unordered_map<schedule::Alloc*, PendingStep*>* most_recent_writers);

void InitSteps(Build* b, const schedule::Schedule& schedule);

class FifoScheduler final : public Scheduler {
 public:
  FifoScheduler(std::size_t alignment, std::uint64_t size_goal, const hal::proto::HardwareSettings& settings);

  schedule::Schedule BuildSchedule(const tile::proto::Program& program, const lang::KernelList& kl) final;

  const char* name() const final;

 private:
  std::size_t alignment_;
  std::uint64_t size_goal_;
  std::uint64_t goal_groups_;
};

// Implements a pre-order traversal of the runnable subset of the steps in a heap of PendingStep.
class RunnableStepsIterator final {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = PendingStep*;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = std::forward_iterator_tag;

  explicit RunnableStepsIterator(std::vector<PendingStep*>* pending = nullptr) : pending_{pending} {
    if (pending_ && (!pending_->size() || pending_->at(0)->dependency_count)) {
      pending_ = nullptr;
    }
  }
  bool operator==(const RunnableStepsIterator& other) const noexcept {
    return (other.pending_ == pending_) && ((pending_ == nullptr) || (other.pos_ == pos_));
  }
  bool operator!=(const RunnableStepsIterator& other) const noexcept { return !(*this == other); }
  PendingStep* operator*() const { return pending_->at(pos_); }
  RunnableStepsIterator& operator++() noexcept;
  RunnableStepsIterator operator++(int) noexcept {
    RunnableStepsIterator res{*this};
    ++*this;
    return res;
  }

 private:
  std::vector<PendingStep*>* pending_;
  std::size_t pos_ = 0;
};

class RunnableSteps final {
 public:
  explicit RunnableSteps(std::vector<PendingStep*>* pending) : pending_{pending} {}
  RunnableStepsIterator begin() const noexcept { return RunnableStepsIterator{pending_}; }
  RunnableStepsIterator end() const noexcept { return RunnableStepsIterator{}; }

 private:
  std::vector<PendingStep*>* pending_;
};

class StepPlan final {
 public:
  StepPlan() : ps_{nullptr} {}
  StepPlan(Build* b, PendingStep* ps);

  void Apply(Build* b);

  operator bool() const { return ps_ != nullptr; }
  bool operator!() const { return ps_ == nullptr; }
  std::size_t mem_needed() const { return mem_needed_; }
  PendingStep* pending_step() const { return ps_; }
  std::uint64_t const input_deltatime_sum() const { return input_deltatime_sum_; }

 private:
  struct LocManip {
    bool add_dep;
    bool is_io;
    schedule::Alloc* contents;
    std::uint64_t delta;
  };

  PendingStep* ps_;
  std::list<Loc> pending_locs_;
  std::vector<Loc*> outputs_;
  std::unordered_map<Loc*, LocManip> used_free_locs_;
  std::size_t mem_needed_ = 0;
  std::list<std::multimap<std::uint64_t, Loc*>::iterator> free_locs_to_mark_as_used_;
  std::uint64_t input_deltatime_sum_ = 0;
};

class StepPlannerIterator final {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = StepPlan;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = std::input_iterator_tag;

  StepPlannerIterator(Build* b, RunnableStepsIterator cur) : b_{b}, cur_{cur} {
    if (cur_ != RunnableStepsIterator{}) {
      plan_ = StepPlan{b_, *cur_};
    }
  }
  StepPlannerIterator() : b_{nullptr}, cur_{} {}

  bool operator==(const StepPlannerIterator& other) const noexcept { return other.cur_ == cur_; }
  bool operator!=(const StepPlannerIterator& other) const noexcept { return other.cur_ != cur_; }
  StepPlan& operator*() { return plan_; }
  StepPlan* operator->() { return &plan_; }
  StepPlannerIterator& operator++() noexcept {
    ++cur_;
    if (cur_ == RunnableStepsIterator{}) {
      plan_ = StepPlan{};
    } else {
      plan_ = StepPlan{b_, *cur_};
    }
    return *this;
  }

 private:
  Build* b_;
  RunnableStepsIterator cur_;
  StepPlan plan_;
};

class StepPlanner final {
 public:
  explicit StepPlanner(Build* b) : b_{b}, steps_{&b->pending} {}
  StepPlannerIterator begin() const noexcept { return StepPlannerIterator{b_, steps_.begin()}; }
  StepPlannerIterator end() const noexcept { return StepPlannerIterator{}; }

 private:
  Build* b_;
  RunnableSteps steps_;
};

}  // namespace fifo_scheduler
}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
