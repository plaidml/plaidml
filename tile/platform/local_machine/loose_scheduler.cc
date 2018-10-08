// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/loose_scheduler.h"

#include <algorithm>
#include <chrono>
#include <iterator>

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

constexpr auto kSchedulerTimeout = std::chrono::seconds(3);

}  // namespace

LooseScheduler::LooseScheduler(const std::shared_ptr<Placer>& placer, std::uint64_t size_goal)
    : placer_{placer}, size_goal_{size_goal} {}

schedule::Schedule LooseScheduler::BuildSchedule(const tile::proto::Program& program, const lang::KernelList& kl) {
  IVLOG(1, "Loose scheduler: attempting to use up to " << size_goal_ << " bytes");

  std::uint_fast32_t broad_loop_count = 0;
  std::uint_fast32_t narrow_loop_count = 0;
  auto end_time = std::chrono::steady_clock::now() + kSchedulerTimeout;

  // The loose scheduling algorithm starts with a linear schedule, in
  // which each step has exactly one synthetic dependency, pointing to
  // the previous step.
  //
  // Next, it adds dataflow dependencies, to guarantee correctness
  // during future schedule adjustments.
  //
  // Then, it performs a broad rescheduling loop: at each step, it
  // decrements each step's synthetic dependency, and then checks the
  // memory usage of the entire schedule, breaking out of the loop
  // (and using the previous iteration's dependencies) if a step
  // exceeds the memory usage goal.
  //
  // After this, it performs a narrow rescheduling loop: almost the
  // same as the broad scheduling loop, but checking the memory usage
  // after adjusting each step, taking back the adjustment if it
  // causes us to exceed the memory cap.  When a particular step
  // causes the memory usage to exceed the cap, that step is removed
  // from consideration for further loosenings.
  //
  // In both loops, if a step's synthetic dependency is subsumed by
  // its required dependencies, we remove the step from consideration
  // for further loosenings.
  //
  // Finally, it runs through the steps: for each, it computes the
  // union of the transitive dependencies of the step's direct
  // dependencies, and then subtracts those dependencies from the
  // step's direct dependencies.  This has no logical effect, but
  // simplifies things a little for the driver and the hardware
  // device, and makes for simpler workflow graphs.
  schedule::Schedule schedule = ToScheduleSteps(program, kl);

  IVLOG(3, "Loose scheduler: original linear schedule is:\n" << schedule);

  if (schedule.steps.size() <= 1) {
    // Just handling the edge case.
    return schedule;
  }

  // Here's what we're going to keep track of, for each candidate step
  // (i.e. steps that we're still shifting dependencies on).
  struct StepInfo {
    schedule::Step* step;                     // The step
    std::list<schedule::Step>::iterator dep;  // The current synthetic dependency
  };

  // Initialize the candidates for loosening.  Note that we skip the
  // first step, since it's never a candidate (it never has program
  // dependencies).  Also, we initialize the synthetic dep of each
  // step to point to itself, since we'll be decrementing these deps
  // as we create the candidate list for each rescheduling pass.
  std::list<StepInfo> candidates;
  auto it = schedule.steps.begin();
  while (++it != schedule.steps.end()) {
    candidates.push_back(StepInfo{&*it, it});
  }

  // Set each step's dependencies to the dependencies required for correct dataflow.
  AddDataflowDeps(&schedule);
  IVLOG(3, "Loose scheduler: dataflow schedule:\n" << schedule);

  // Reserve a variable to be the selected placement, and reserve a
  // variable for the synthetic deps that produced this placement.
  std::unique_ptr<Placement> placement;

  // Execute the broad rescheduling loop.
  bool reached_timeout = false;
  while (!placement || candidates.size()) {
    if (placement && (end_time < std::chrono::steady_clock::now())) {
      LOG(WARNING) << "Reached scheduler optimization timeout";
      reached_timeout = true;
      break;
    }
    ++broad_loop_count;
    std::list<StepInfo> new_candidates;

    // Build new_candidates from the synthetic dependencies in
    // candidates.
    for (StepInfo candidate : candidates) {
      if (candidate.dep == schedule.steps.begin()) {
        // Omit this candidate; the previous version already had a
        // synthetic dependency on the first step, so the updated
        // version is to have no synthetic dependency at all.
        continue;
      }
      --candidate.dep;
      auto res = candidate.step->deps.emplace(&*candidate.dep);
      if (!res.second) {
        // The step already had this dep in its dependencies -- so the
        // updated synthetic dependency is actually a real data
        // dependency.  There's no point in including this synthetic
        // dependency in further trials, and there's no point in
        // continuing to consider this step for synthetic
        // dependencies.
        continue;
      }
      // Save the candidate, so that we remember that the new
      // placement included this candidate's synthetic dependency.
      new_candidates.emplace_back(candidate);
    }

    // Create new_placement based on new_candidates.
    auto new_placement = placer_->PlaceSchedule(program, &schedule);

    IVLOG(4, "Loose scheduler: trial broad schedule uses " << new_placement->device_memory_bytes() << " bytes:\n"
                                                           << schedule);

    // Remove the synthetic deps specified by new_candidates; these
    // were the synthetic deps earlier added in order to create the
    // new placement.
    for (auto& candidate : new_candidates) {
      candidate.step->deps.erase(&*candidate.dep);
    }

    // If we already have a placement, and the new one exceeds the
    // memory cap, we can't use the new one; stick with what we have.
    if (placement && size_goal_ < new_placement->device_memory_bytes()) {
      break;
    }

    // Save what we have so far; it's either the first placement or
    // it's better than the best placement we've found so far.
    placement = std::move(new_placement);
    candidates.clear();
    candidates.swap(new_candidates);
  }

  // Add the current synthetic dependency candidates back into the
  // schedule.  These are the synthetic dependencies that were used to
  // create the current placement; we know they're valid and don't
  // equal any current dependencies for any given step.
  for (StepInfo& candidate : candidates) {
    candidate.step->deps.emplace(&*candidate.dep);
  }

  IVLOG(3, "Loose scheduler: broad schedule is:\n" << schedule);

  // Execute the narrow scheduling loop.
  while (candidates.size() && !reached_timeout) {
    if (end_time < std::chrono::steady_clock::now()) {
      LOG(WARNING) << "Reached scheduler optimization timeout";
      reached_timeout = true;
      break;
    }
    ++narrow_loop_count;
    auto current = candidates.begin();
    while (current != candidates.end() && std::chrono::steady_clock::now() < end_time) {
      // Try moving this candidate's dependency back a step, and
      // seeing what that does to the temporary allocations.
      auto& candidate = *current;
      candidate.step->deps.erase(&*candidate.dep);
      bool using_trial_dep = false;
      auto trial_dep = candidate.dep;
      if (candidate.dep != schedule.steps.begin()) {
        --trial_dep;
        using_trial_dep = candidate.step->deps.emplace(&*trial_dep).second;
      }
      auto new_placement = placer_->PlaceSchedule(program, &schedule);
      IVLOG(4, "Loose scheduler: trial narrow schedule uses " << new_placement->device_memory_bytes() << " bytes:\n"
                                                              << schedule);

      bool keep_candidate;

      if (placement->device_memory_bytes() < new_placement->device_memory_bytes() &&
          size_goal_ < new_placement->device_memory_bytes()) {
        // Well, that didn't work out: the new placement is past our
        // size goal, and is using more memory than the previous
        // placement.  So let's put the dependency back where it was,
        // and stop considering this candidate.
        if (using_trial_dep) {
          candidate.step->deps.erase(&*trial_dep);
        }
        candidate.step->deps.emplace(&*candidate.dep);
        keep_candidate = false;
      } else {
        // Either the new placement uses the same amount of memory (or
        // less, although that shouldn't ever happen), or it uses
        // more, but still fits within our size goal.  Either way, it
        // has an earlier dependency, and is fine for use.
        placement = std::move(new_placement);

        if (using_trial_dep) {
          // Update the candidate dep, and remember to try pushing it
          // back when we come around the candidate list again.
          candidate.dep = trial_dep;
          keep_candidate = true;
        } else {
          // Not using a synthetic dep was okay, but we're done
          // considering this candidate.
          keep_candidate = false;
        }
      }

      auto rm = current;
      ++current;

      if (!keep_candidate) {
        candidates.erase(rm);
      }
    }
  }

  IVLOG(3, "Loose scheduler: loose schedule is:\n" << schedule);

  // Remove implied dependencies.
  std::vector<std::set<schedule::Step*>> transitive_deps{schedule.steps.size()};
  for (auto& step : schedule.steps) {
    auto& tdeps = transitive_deps[step.idx];
    std::set<schedule::Step*> sdeps;
    sdeps.swap(step.deps);
    for (auto dep : sdeps) {
      tdeps.insert(transitive_deps[dep->idx].begin(), transitive_deps[dep->idx].end());
    }
    std::set_difference(sdeps.begin(), sdeps.end(), tdeps.begin(), tdeps.end(),
                        std::inserter(step.deps, step.deps.end()));
    std::copy(step.deps.begin(), step.deps.end(), std::inserter(tdeps, tdeps.end()));
  }

  // Apply the placement, generating the final schedule.
  placement->Apply();
  IVLOG(1, "Loose scheduler: scheduled with " << broad_loop_count << " broad loop(s) and " << narrow_loop_count
                                              << " narrow loop(s)");
  IVLOG(2, "Loose scheduler: final schedule is:\n" << schedule);
  return schedule;
}

const char* LooseScheduler::name() const { return "Loose"; }

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
