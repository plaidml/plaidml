// Copyright 2018, Intel Corporation.

#include "tile/platform/local_machine/fifo_scheduler.h"

#include <vector>

#include "base/util/error.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// The FIFO scheduler algorithm makes the following assumptions:
//
// * A device is a group of homogenous compute units pulling work groups from a
//   limited-lookahead out-of-order queue.
//
// * Inputs will start out on the device (this might change in the future).
//
// * Outputs should end up on the device (this might change in the future).
//
// The algorithm attempts to order steps such that if step S is running, step S+1 will not depend
// on the completion of S, allowing S+1 to start executing as soon as compute units become
// available, minimizing the impact of straggling work groups.  The ordering also attempts to take
// device caches into account, attempting to schedule consumers to run shortly after producers.
//
// The FIFO scheduler algorithm is:
//   * Pending : set of Step; initialized to all program steps
//   * Runnable : set of Step
//   * CopyFromHost : set of Step  // TODO
//   * Retired: list of Step; building up the final schedule
//   * Running: list of Step
//   * FreeLocs: set of Loc; initialized to the worst-case loc set.
//               Note that locs may contain usable values even though they're in
//               FreeLocs, e.g. if there's an on-host copy of the loc's value.
//
//   * While there are steps in Pending or Runnable:
//     * Move steps from Pending to Runnable if all of their dependencies are satisfied:
//       if all of their non-program-input dependencies are accounted for in Retired.
//
//     * Schedule as many Runnable steps as possible, moving them from Runnable to Running.
//       A Runnable step may be scheduled if there's enough memory for its outputs and all
//       of its inputs are present on its device; output memory gets pulled from FreeLocs
//       and locs are created as necessary (see below for details).
//
//     * If there are steps that are Runnable but for lack of output memory, and there are
//       locs in FreeLocs that are too small, attempt to increase the size of the largest
//       loc in FreeLocs to make it work for some step in Runnable, and schedule that step.
//
//     * If there's a step that's Runnable except that its inputs are on-host, and there
//       are free locs of the right size available for the step's inputs, insert copy steps
//       at the points where those locs became available.  (Note that while this will make
//       the step Pending instead of Runnable, the step is guaranteed to become Runnable
//       again before we will consider inserting steps to free up the loc inputs, so we're
//       guaranteed to make forward progress).
//
//     * If there are no steps in Runnable, no locs in FreeLocs, or not enough memory
//       to sufficiently increase the size of any FreeLocs, and there are steps in Running,
//       retire steps from the front of the Running queue to attempt to alleviate the problem
//       (by advancing the iterator separating Retired from Running).  For all input values that
//       are only accessed by kernels in Retired, move the underlying loc to FreeLocs.
//
//     * TODO: Swapping to host.
//
//       If there are no steps in Running, and there are steps in Runnable, we must be bound
//       by memory constraints (no locs in FreeLocs and not enough memory to create a new loc
//       of the right size, or not enough memory to sufficiently expand any locs in FreeLocs),
//
//       To handle this:
//         * Select a step in Runnable; locate from FreeLocs for as many outputs as possible.
//
//         * For each output that still needs an loc:
//           * From the outstanding locs that aren't used by the step, and are big enough for
//             the current output (or that can be expanded given device memory constraints):
//               Select the loc that's next used as far as possible after this step.
//
//           * Insert a copy-to-host step at the point where the step that wrote that loc
//             was retired.
//
//           * If there's an unused loc smaller than the value's current loc and big enough to
//             contain the value, insert a copy-local step at the point where that loc became
//             available.  This is guaranteed to be some point after the step that wrote the value
//             was scheduled to run, because otherwise we would have selected the new loc in the
//             first place at the point where that step was scheduled.
//
//   * TODO: Swapping to host.
//
//     Perform extraneous copy removal: if an loc is written by a copy but then that value is
//     subsequently unused, prune the copy.
//
//   * Finally: Turn the retired step list into a Schedule, and perform transitive dependency
//     elimination.
//
// Scheduling steps naively may create locs that would cause subsequent steps to become
// unschedulable -- too much memory could be used in the wrong arrangement of locs for a
// particular kernel to ever be runnable.  To support this when using block placement, we first
// build up the worst-case loc set: the minimum set of minimally-sized locs needed in order
// to run every kernel in the program, taking into account consumable input buffers and all output
// buffers.  We add these locs to the free loc list at the start of scheduling, guaranteeing
// that in the worst possible case, for any step, it's possible for the schedule to swap all values
// out to the host, swap temporaries back in to minimally-sized locs, assign the outputs to
// minimally-sized locs, and run the step.
//
// It's possible that this will still be insufficient -- the worst-case loc set may be too large
// to be scheduled.  To handle this, we'll need to split kernels; implementing that is left as
// TBD.  For now, we assume that the worst-case loc set can in fact be allocated, and thus the
// entire program can be scheduled.
//
// When selecting memory to use for step outputs, there will be times when reusing an existing loc
// and creating a new loc are both reasonable paths.  To resolve this:
//
// * Program outputs are restricted to equal-size locs; with that in mind, they can be treated
//   identically to intraprogram locs.  (TODO: We want to have this restriction so that inputs
//   and outputs always end up being exactly the right size.  We could also implement this by
//   continuing to track program input locs (which cannot be resized) and inserting copy steps
//   to ensure that program outputs wind up in correctly-sized locs.  We should measure to see
//   whether it's worth the copies to make this happen.)
//
// * If there's an existing free loc, reuse it.  Choose the smallest loc that fits the value.
//
//   If there's an in-use loc that's sufficient for the current value and occupied by a smaller
//   value, and if there's a free loc that was free when the smaller value was created,
//   swap them: use the larger free loc for the earlier, smaller value, and use the smaller,
//   better-fitting loc for the current, larger value.
//
//   The goal here: if a value needs to be swapped to system memory or needs to have a new loc
//   created to hold it, we want to be swapping/placing the smaller value.
//
// * Create a new loc, if memory permits.
//
//   If there's an in-use loc that's sufficient for the current value and occupied by a smaller
//   value, create a new loc for the smaller value, and use the in-use loc for the current
//   value.
//
// Datastructures:
//
// There are several Run step states:
// Pending: The step has unsatisfied data dependencies.
// Runnable: The step's data dependencies have been scheduled, and the step can be considered for
//           scheduling.
// Running: The step's scheduled to run.
// Retired: The step's known to have finished running.
//
// We handle Pending and Runnable by maintaining a binary heap of pointers to PendingStep structs,
// with the next steps to be made Runnable at the front of the heap (note that steps in Runnable
// may not actually be schedulable due to memory constraints, so we'll still need the ability to
// walk all steps in Runnable).  We use a bespoke heap implementation built on std::vector, so that
// we can maintain the contents' self-awareness of where they are in the heap and perform
// key manipulation and traversal operations that aren't provided by the STL.
//
// For Retired and Running, we use a std::list of ScheduledStep structs, and use iterators to
// keep track of where the algorithm is in the list -- the line between Retired and Running.
//
// For each value, we keep track of where it currently is relative to the retired steps.
// Note that once swapping has been implemented, a value may have up to two locations at any given
// time: one on-host location and one retired on-device location.  But at any given point, there
// will be exactly one canonical on-device location that steps should be pulling values from.

namespace fifo_scheduler {

// TODO: Make this a configurable parameter.
// This should probably be related to the size of the device's L2 cache, given that the tile size
// is already selected for L1 cache performance.
constexpr std::uint64_t kMaxInputDeltatime = 100 * std::kilo::num;

// Used to define the heap ordering for the pending-step heap.
bool PendingStepHeapLess(const PendingStep* lhs, const PendingStep* rhs) {
  // The requirement is that steps with no pending dependencies must come before steps that have
  // dependencies.  Since std:: heaps are max-heaps, an element lhs always "less than" an element
  // rhs if lhs has more outstanding dependencies:
  return lhs->dependency_count > rhs->dependency_count;
}

// Rounds a byte size up to the next divisible-by-alignment value.
std::uint64_t AlignUp(Build* b, std::uint64_t byte_size) {
  return ((byte_size + b->alignment - 1) / b->alignment) * b->alignment;
}

// Adds a synthetic output-consuming step to a schedule.
void PushSyntheticFinalOutputStep(Build* b, schedule::Schedule* schedule, const tile::proto::Program& program) {
  schedule::Step step{schedule::Step::Tag::kRun};
  step.idx = schedule->steps.back().idx + 1;
  for (auto& alloc : schedule->allocs) {
    if (alloc.is_tmp()) {
      continue;
    }
    if (alloc.is_input() && program.inputs().at(alloc.input).consumed()) {
      continue;
    }
    step.inputs.push_back(&alloc);
    // Start the IO allocs with a refcount of 1, so that their contents will be valid even after
    // the final output step has been processed.
    b->alloc_refcounts.emplace(&alloc, 1);
  }
  schedule->steps.emplace_back(std::move(step));
}

// Adds a new PendingStep to the build.
PendingStep* NewPendingStep(Build* b, const schedule::Step* step) {
  auto kidx = step->kidx;
  bool is_zero = (kidx < b->kl->kernels.size() && b->kl->kernels[kidx].ktype == lang::KernelType::kZero &&
                  step->inputs.size() == 0 && step->outputs.size() == 1);
  b->pending_steps_storage.emplace_back(PendingStep{0, step, 0, is_zero});
  return &b->pending_steps_storage.back();
}

// Convert a PendingStep into a ScheduledStep, removing the PendingStep from the pending/runnable
// step heap (although leaving its storage intact; the step may still be safely accessed).
// The ScheduledStep will point to the PendingStep's Step; the PendingStep's dependents
// will be moved to the ScheduledStep.
ScheduledStep* MovePendingStepToScheduled(Build* b, PendingStep* ps) {
  // Advance memtime by the size of the step's inputs.
  for (auto* input : ps->step->inputs) {
    b->current_memtime += b->value_locs[input]->byte_size;
  }
  // Mark inputs as being loaded as of this time.
  for (auto* input : ps->step->inputs) {
    b->value_locs[input]->cache_memtime = b->current_memtime;
  }
  auto ss =
      b->scheduled.emplace(b->scheduled.end(), ScheduledStep{ps->step, ps->work_groups, std::move(ps->dependents)});
  if (b->running == b->scheduled.end()) {
    --b->running;
  }
  b->running_groups += ss->work_groups;
  RemovePendingStep(&b->pending, ps);
  return &*ss;
}

void TryRetireScheduledStep(Build* b) {
  if (b->running == b->scheduled.end()) {
    return;
  }

  // Retire the step at the front of the running list, completing its outputs' dependencies
  // and possibly making its inputs' allocs available for re-use.
  for (PendingStep* dep : b->running->dependents) {
    ResolveDep(&b->pending, dep);
  }
  b->running->dependents.clear();
  for (Loc* loc : b->running->outputs) {
    b->value_locs.emplace(loc->contents, loc);
  }
  for (schedule::Alloc* alloc : b->running->step->inputs) {
    auto ait = b->alloc_refcounts.find(alloc);
    if (ait == b->alloc_refcounts.end()) {
      LOG(FATAL) << "Unable to find alloc " << alloc << " in alloc_refcounts";
    }
    if (!--ait->second) {
      auto lit = b->value_locs.find(alloc);
      if (lit == b->value_locs.end()) {
        LOG(FATAL) << "Unable to find alloc " << alloc << " in value_locs";
      }
      b->free_locs.emplace(lit->second->byte_size, lit->second);
      b->value_locs.erase(lit);
    }
  }
  // Advance memtime by the size of the step's outputs.
  for (schedule::OutputInfo oi : b->running->step->outputs) {
    b->current_memtime += b->value_locs[oi.allocp]->byte_size;
  }
  // Mark outputs as being loaded as of this time.
  for (schedule::OutputInfo oi : b->running->step->outputs) {
    b->value_locs[oi.allocp]->cache_memtime = b->current_memtime;
  }
  b->running_groups -= b->running->work_groups;
  ++b->running;
}

void ReduceAvailableMem(Build* b, std::uint64_t mem_used) { b->mem_available -= std::min(b->mem_available, mem_used); }

// Splices the supplied uncommitted locs into the committed locs list, decrementing the amount of
// memory available by the requested amount.
void SpliceLocs(Build* b, std::list<Loc> pending_locs, std::uint64_t mem_used) {
  b->locs.splice(b->locs.end(), std::move(pending_locs));
  ReduceAvailableMem(b, mem_used);
}

// Marks the indicated free loc as being in-use.
void MarkLocInUse(Build* b, std::multimap<std::uint64_t, Loc*>::iterator loc_to_remove) {
  b->free_locs.erase(loc_to_remove);
}

void InitStep(Build* b, const schedule::Step& step,
              std::unordered_map<schedule::Alloc*, PendingStep*>* most_recent_writers) {
  auto* ps = NewPendingStep(b, &step);

  // Build up the step's allocs.
  std::vector<schedule::Alloc*> allocs;
  allocs.reserve(step.inputs.size() + step.outputs.size());

  {
    // While building the allocs vector, we only add outputs if we don't see aliasable inputs.
    // So: keep track of which step input allocs have not yet been reused for step output allocs.
    std::unordered_set<schedule::Alloc*> reusable_input_allocs;
    reusable_input_allocs.reserve(step.inputs.size());

    for (schedule::Alloc* alloc : step.inputs) {
      allocs.emplace_back(alloc);
      if (!alloc->is_input()) {
        ps->dependency_count++;
        most_recent_writers->at(alloc)->dependents.emplace_back(ps);
      }
      reusable_input_allocs.insert(alloc);
      auto res = b->alloc_refcounts.emplace(alloc, 1);
      if (!res.second) {
        res.first->second++;
      }
    }
    for (const schedule::OutputInfo& oi : step.outputs) {
      auto mrw_ins = most_recent_writers->emplace(oi.allocp, ps);
      if (!mrw_ins.second) {
        // This output is also the output of another step (since there was an existing entry in the
        // most_recent_writers map).  So this is presumably a partial update of the underlying
        // data.  We need to take a dependency on the earlier writer, and we always need to find
        // a worst-case alloc for it.
        if (mrw_ins.first->second->is_zero) {
          ps->zero_inputs.emplace_back(mrw_ins.first->second, oi.allocp);
        } else {
          ps->dependency_count++;
          mrw_ins.first->second->dependents.emplace_back(ps);
        }
        mrw_ins.first->second = ps;  // Set this step as the most-recent-writer.
      } else {
        bool found = false;
        // See if we can match up this step output with a step input.
        for (schedule::Alloc* can_alias : oi.allocp->safe_self_alias_allocs) {
          auto it = reusable_input_allocs.find(can_alias);
          if (it != reusable_input_allocs.end()) {
            reusable_input_allocs.erase(it);
            found = true;
            break;
          }
        }
        if (!found) {
          // We'll need a new alloc for this program output.
          allocs.emplace_back(oi.allocp);
        }
      }
    }
  }

  // Sort the allocs
  std::sort(allocs.begin(), allocs.end(), [](schedule::Alloc* lhs, schedule::Alloc* rhs) {
    // Largest allocs first
    if (lhs->byte_size > rhs->byte_size) {
      return true;
    }
    if (lhs->byte_size < rhs->byte_size) {
      return false;
    }
    // Then, sort intra-program allocs ahead of program inputs and outputs
    bool ltmp = lhs->is_tmp();
    bool rtmp = rhs->is_tmp();
    if (ltmp && !rtmp) {
      return true;
    }
    if (rtmp && !ltmp) {
      return false;
    }
    // Finally, just use the alloc addresses as a tiebreaker.
    return lhs < rhs;
  });

  // For each alloc, validate that there's an Loc big enough for the alloc.
  auto lit = b->locs.begin();
  for (schedule::Alloc* alloc : allocs) {
    if (lit == b->locs.end()) {
      // Add a new AllocInfo.
      lit = b->locs.emplace(lit, Loc{alloc->byte_size, !alloc->is_tmp()});
    } else if (alloc->is_tmp()) {
      if (lit->byte_size < alloc->byte_size) {
        if (!lit->is_io) {
          // Increase the size of the current loc.  This is always cheaper than adding a brand-new
          // Loc; it's possible that the current Loc is just right for some subsequent
          // Alloc, but if so, we'll just allocate a new Loc of the correct size when we see
          // that Alloc.
          lit->byte_size = alloc->byte_size;
        } else {
          // Since the current Loc is too small but needs to stay its current size, create a
          // new Loc.
          lit = b->locs.emplace(lit, Loc{alloc->byte_size});
        }
      }
      // Otherwise, this pure-temporary alloc can happily fit into the equal-or-larger Loc.
    } else {
      // This is a program input or output alloc, and it needs to have an exactly-matching Loc.
      while ((lit != b->locs.end()) && (alloc->byte_size < lit->byte_size)) {
        ++lit;
      }
      if ((lit != b->locs.end()) && (alloc->byte_size == lit->byte_size)) {
        // We can use this Loc for this IO alloc.
        lit->is_io = true;
      } else {
        // We need a new Loc in order to match the size.
        lit = b->locs.emplace(lit, Loc{alloc->byte_size, true});
      }
    }
    ++lit;
  }
}

void InitSteps(Build* b, const schedule::Schedule& schedule) {
  std::unordered_map<schedule::Alloc*, PendingStep*> most_recent_writers;
  for (const auto& step : schedule.steps) {
    InitStep(b, step, &most_recent_writers);
  }

  // Add the distance-from-outputs measurement.
  for (auto sit = b->pending_steps_storage.rbegin(); sit != b->pending_steps_storage.rend(); ++sit) {
    std::uint64_t distance = 0;
    for (auto dep : sit->dependents) {
      distance = std::max(distance, dep->distance);
    }
    auto kidx = sit->step->kidx;
    std::uint64_t kdist;
    if (kidx < b->kl->kernels.size() && !sit->is_zero) {
      const lang::KernelInfo& kinfo = b->kl->kernels[kidx];
      kdist = kinfo.info.perf_stats().work_groups();
    } else {
      // TODO: Update the test code to provide a synthetic KernelInfo.
      kdist = 1;
    }
    distance += kdist;
    sit->work_groups = std::min(kdist, b->work_group_limit);
    sit->distance = distance;
  }
}

std::vector<PendingStep*> InitPendingSteps(std::list<PendingStep>* pending_steps_storage) {
  std::vector<PendingStep*> pending;
  pending.reserve(pending_steps_storage->size());
  for (PendingStep& ps : *pending_steps_storage) {
    if (!ps.is_zero) {
      pending.emplace_back(&ps);
    }
  }
  std::make_heap(pending.begin(), pending.end(), PendingStepHeapLess);
  std::size_t loc = 0;
  for (auto& psp : pending) {
    psp->loc = loc++;
  }
  return pending;
}

void RemovePendingStep(std::vector<PendingStep*>* pending, PendingStep* ps) {
  // If step happens to be at the end of the heap, we can just shrink the heap
  // and be done.  Note that this handles the last-item-in-the-heap case.
  if (ps->loc == (pending->size() - 1)) {
    pending->resize(pending->size() - 1);
    return;
  }

  // Logically remove the requested step by recursively bubbling parents downward to fill in the
  // gap in the heap left by the removal, logically moving the requested step to location 0.
  std::size_t loc = ps->loc;
  while (loc) {
    std::size_t ploc = (loc - 1) >> 1;
    ps = pending->at(ploc);
    ps->loc = loc;
    (*pending)[loc] = ps;
    loc = ploc;
  }

  // Move the final item in the heap to the root, shrinking the heap.
  ps = pending->back();
  ps->loc = 0;
  pending->resize(pending->size() - 1);

  // Bubble the step downward to preserve the max-heap property w.r.t. PendingStepHeapLess
  // (which considers the greatest step to be one with no dependencies).
  for (;;) {
    std::size_t lc = (loc << 1) + 1;
    std::size_t rc = lc + 1;
    PendingStep* left = nullptr;
    PendingStep* right = nullptr;
    bool is_ge_children = true;
    if (lc < pending->size()) {
      left = pending->at(lc);
      is_ge_children = !PendingStepHeapLess(ps, left);
      if (rc < pending->size()) {
        right = pending->at(rc);
        is_ge_children = is_ge_children && !PendingStepHeapLess(ps, right);
      }
    }
    if (is_ge_children) {
      // No swap needed; the removal is complete.
      (*pending)[loc] = ps;
      return;
    }
    PendingStep* swap = left;  // Assume swap with left child
    std::size_t sc = lc;
    if (right && PendingStepHeapLess(left, right)) {
      // Swap parent and right child.
      swap = right;
      sc = rc;
    }
    swap->loc = loc;
    (*pending)[loc] = swap;
    ps->loc = sc;
    loc = sc;
  }
}

// Returns true iff lhs is a better plan than rhs.
bool IsBetterPlan(Build* b, const StepPlan& lhs, const StepPlan& rhs) {
  // A plan below the memory limit is axiomatically better than one above; a plan that
  // exceeds the budget by a smaller amount is better than one that exceeeds the budget by a
  // larger amount.
  if (b->mem_available < rhs.mem_needed()) {
    if (lhs.mem_needed() <= b->mem_available) {
      return true;
    }
    if (lhs.mem_needed() < rhs.mem_needed()) {
      return true;
    }
  } else if (b->mem_available < lhs.mem_needed()) {
    return false;
  }

  // Steps whose inputs are in cache are better than steps whose inputs are not in cache.
  if (lhs.input_deltatime_sum() < rhs.input_deltatime_sum()) {
    return true;
  }
  if (rhs.input_deltatime_sum() < lhs.input_deltatime_sum()) {
    return false;
  }

  // Otherwise, the best plan is the one whose step is the greatest distance from completion.
  return lhs.pending_step()->distance > rhs.pending_step()->distance;
}

// Attempt to schedule at least one runnable step, returning true iff a step was scheduled.
bool ScheduleRunnableStep(Build* b) {
  StepPlan best;
  for (StepPlan& plan : StepPlanner{b}) {
    if (b->running != b->scheduled.end()) {
      // Not everything has been retired, so we want a couple of additional checks in order
      // to prioritize useful work.
      if (b->mem_available < plan.mem_needed()) {
        // Don't bother with over-the-limit plans.
        continue;
      }
      if (kMaxInputDeltatime < plan.input_deltatime_sum()) {
        // Don't bother with plans whose inputs are ancient.
        continue;
      }
    }
    if (!best || IsBetterPlan(b, plan, best)) {
      best = std::move(plan);
    }
  }
  if (!best) {
    return false;
  }
  best.Apply(b);
  return true;
}

// Adjusts for the fact that one dependency of a PendingStep has been retired.
void ResolveDep(std::vector<PendingStep*>* pending, PendingStep* ps) {
  ps->dependency_count--;

  // Lowering the dependency count can't cause the pending step to have a higher dependency count
  // than its children, but it can cause the pending step to have a lower dependency count than its
  // parents.  To restore the heap property, we need to bubble the step towards the root of the
  // heap; to slightly prioritize steps that were recently made runnable, we bubble the step as far
  // upwards as possible.  (Note that the heap is a max-heap; greater heap elements have a lower
  // dependency count.)
  std::size_t loc = ps->loc;
  while (loc) {
    std::size_t ploc = (loc - 1) >> 1;
    PendingStep* parent = pending->at(ploc);
    if (PendingStepHeapLess(ps, parent)) {
      // The step cannot be swapped with its parent.
      break;
    }
    parent->loc = loc;
    (*pending)[loc] = parent;
    loc = ploc;
  }
  ps->loc = loc;
  (*pending)[loc] = ps;
}

// Adds dataflow dependencies to a schedule.
void AddDeps(schedule::Schedule* schedule) {
  // TODO: Consider replacing AddDataflowDeps with this version.  They should perform identically for the
  // existing schedulers; the only goal of having a separate implementation is to ensure that changes
  // made for FifoScheduler don't affect existing schedulers.
  struct BusyInfo {
    schedule::Step* latest_writer;
    std::unordered_set<schedule::Step*> active_readers;
  };
  std::unordered_map<schedule::Alloc*, BusyInfo> busy_infos;
  std::vector<std::set<schedule::Step*>> transitive_deps{schedule->steps.size()};
  for (auto& step : schedule->steps) {
    std::set<schedule::Step*> deps;
    IVLOG(3, "Adding dataflow deps to s" << step.idx);
    for (schedule::Alloc* allocp : step.inputs) {
      if (!allocp->byte_size) {
        continue;
      }
      IVLOG(3, "  Getting input deps for a" << allocp->idx);
      auto bit = busy_infos.emplace(allocp, BusyInfo{nullptr}).first;
      if (!bit->second.latest_writer) {
        if (!allocp->is_input()) {
          std::stringstream ss;
          ss << "Program fails to initialize non-empty temporary for a" << allocp->idx;
          throw error::Internal(ss.str());
        }
        IVLOG(3, "  a" << allocp->idx << " is input \"" << allocp->input << "\"; no deps needed");
      } else {
        IVLOG(3, "  Adding dep on a" << allocp->idx << " with last writer s" << bit->second.latest_writer->idx);
        deps.insert(bit->second.latest_writer);
      }
      bit->second.active_readers.emplace(&step);
    }
    for (schedule::OutputInfo oi : step.outputs) {
      schedule::Alloc* allocp = oi.allocp;
      IVLOG(3, "  Adding output dep on a" << allocp->idx);
      auto res = busy_infos.emplace(allocp, BusyInfo{&step});
      if (res.second) {
        IVLOG(3, "    This was the first writer");
        // This was the first insertion of this alloc; everything is good.
        continue;
      }
      if (res.first->second.latest_writer) {
        // The alloc already existed in the map, and had a writer -- so some other step has updated
        // part of the alloc.  The current step has a dependency on that previous writer, and all
        // current readers.
        IVLOG(3, "    Alloc had been written by s" << res.first->second.latest_writer->idx);
        deps.insert(res.first->second.latest_writer);
      }
      deps.insert(res.first->second.active_readers.begin(), res.first->second.active_readers.end());

      // The current step becomes the latest writer, and current readers don't matter anymore.
      res.first->second.latest_writer = &step;
      res.first->second.active_readers.clear();
    }
    std::set<schedule::Step*>& tdeps = transitive_deps[step.idx];
    for (schedule::Step* depstep : deps) {
      tdeps.insert(transitive_deps[depstep->idx].begin(), transitive_deps[depstep->idx].end());
    }

    std::set_difference(deps.begin(), deps.end(), tdeps.begin(), tdeps.end(),
                        std::inserter(step.deps, step.deps.end()));

    for (schedule::Step* dep : deps) {
      tdeps.insert(dep);
    }
  }
}

// Turn the scheduled steps into a runnable schedule.
schedule::Schedule FinalizeSchedule(Build* b) {
  schedule::Schedule result;
  std::unordered_map<schedule::Alloc*, schedule::Alloc*> alloc_allocs;
  std::unordered_map<Loc*, schedule::Alloc*> loc_allocs;
  for (Loc& loc : b->locs) {
    if (!loc.contents) {
      // The loc wasn't actually needed (could've been precautionary); ignore it.
      continue;
    }
    schedule::Alloc alloc;
    alloc.byte_size = loc.byte_size;
    alloc.input = loc.input;
    alloc.output = loc.output;
    auto ait = result.allocs.emplace(result.allocs.end(), std::move(alloc));
    loc_allocs[&loc] = &*ait;
  }

  for (auto kvp : b->input_locs) {
    alloc_allocs[kvp.first] = loc_allocs[kvp.second];
  }

  for (ScheduledStep& ss : b->scheduled) {
    schedule::Step step{ss.step->tag};
    for (schedule::Alloc* input : ss.step->inputs) {
      step.inputs.emplace_back(alloc_allocs.at(input));
    }
    auto step_output_it = ss.step->outputs.begin();
    for (Loc* output : ss.outputs) {
      schedule::Alloc* output_alloc = loc_allocs.at(output);
      step.outputs.emplace_back(schedule::OutputInfo{output_alloc, output->add_dep});
      alloc_allocs[step_output_it->allocp] = output_alloc;
      ++step_output_it;
    }
    step.kidx = ss.step->kidx;
    result.steps.emplace_back(std::move(step));
  }

  // Pop the synthetic output-consuming step from the schedule.
  result.steps.pop_back();

  result.Reindex();

  // TODO: For caching reasons, it may be more optimal to include additional synthetic dependencies
  // in the final schedule, by making scheduled steps depend transitively on everything in the
  // retired set as of the time when they're scheduled.  To do that, we'd need to have additional
  // synthetic dependencies, and we'd need to supply them to AddDeps() (or add them after calling
  // AddDeps()).

  AddDeps(&result);
  return result;
}

FifoScheduler::FifoScheduler(std::size_t alignment, std::uint64_t size_goal,
                             const hal::proto::HardwareSettings& settings)
    : alignment_{alignment}, size_goal_{size_goal}, goal_groups_{settings.goal_groups()} {
  if (goal_groups_ < 2) {
    goal_groups_ = 2;  // Just to pick something.
  }
}

schedule::Schedule FifoScheduler::BuildSchedule(const tile::proto::Program& program, const lang::KernelList& kl) {
  schedule::Schedule start = ToScheduleSteps(program, kl);
  IVLOG(4, "Scheduling program:\n" << program.code());
  IVLOG(3, "Initial schedule:\n" << start);

  Build b{program, kl, start.steps.size(), alignment_, size_goal_, goal_groups_};

  // Ensure that all outputs and inputs are in GPU allocs.
  PushSyntheticFinalOutputStep(&b, &start, program);

  // Process the steps into PendingSteps, recording which step uses which alloc and ensuring
  // that a minimal set of allocs is available s.t. every step is able to run.
  InitSteps(&b, start);

  // Add the input allocs after walking the steps, so that they're not considered part of the
  // minimal alloc set -- so we can skip considering them for swapping and still preserve
  // forward progress.  (Alternatively, we could add them before processing the steps, so
  // that they could be part of the minimal set, but then we'd have to be able to swap them,
  // and we'd have to either be sure to bring them back to their original allocs or we'd
  // need to extend the schedule definition a bit to allow for renames.  It's simplest to
  // just keep them separate for now.)
  for (auto& alloc : start.allocs) {
    if (alloc.is_input()) {
      auto lit = b.locs.emplace(b.locs.end(), Loc{alloc.byte_size, true});
      lit->contents = &alloc;
      lit->input = alloc.input;
      b.input_locs[&alloc] = &*lit;
    }
  }

  // Sum the memory used so far, and add the initial Locs to the loc tracking maps.
  for (Loc& loc : b.locs) {
    ReduceAvailableMem(&b, AlignUp(&b, loc.byte_size));
    if (loc.contents) {
      b.value_locs.emplace(loc.contents, &loc);
    } else {
      b.free_locs.emplace(loc.byte_size, &loc);
    }
  }

  IVLOG(1, "Minimal loc count: " << b.locs.size() << " Remaining mem: " << b.mem_available);

  // Heapify the pending steps.
  b.pending = InitPendingSteps(&b.pending_steps_storage);

  // Process the heap.
  while (b.pending.size()) {
    while (ScheduleRunnableStep(&b)) {
      auto pending_groups = b.running_groups - b.running->work_groups;
      if (goal_groups_ <= pending_groups) {
        break;
      }
    }

    TryRetireScheduledStep(&b);
  }

  // Assign output locs.
  for (auto& alloc : start.allocs) {
    if (alloc.is_output()) {
      b.value_locs[&alloc]->output = alloc.output;
    }
  }

  // TODO: Anneal the schedule.
  //
  // At this point, we have a valid schedule: we've chosen a topological ordering for the steps
  // and selected memory allocations s.t. that fit within the device's available memory
  // (if that's possible).
  //
  // The schedule may not use all of the device's memory -- the fifo scheduling algorithm tries to
  // reuse memory conservatively in scheduling, since it doesn't revisit scheduling decisions and
  // may need that memory later on. This can cause points in the schedule where we're reusing
  // memory more aggressively than we need to -- places where we could use additional memory,
  // giving the hardware the option to run a few more kernels in parallel.  We can think of the
  // schedule as being "stressed" at these points, and call the process of removing those stresses
  // "annealing".
  //
  // So we should anneal the schedule: analyze the existing scheduled steps, identifying and
  // ranking stresses (where temporally close synthetic dependencies are considered higher
  // priority), and then inserting additional memory locs to relax stresses where possible.
  //
  // N.B.:
  //   * The step planner resizes existing free locs before adding additional locs.  So there won't
  //     be locs created for future outputs that could have been created for earlier outputs.  But
  //     given three outputs -- O1, O2, and O3, created in that order -- O1 and O2 might show use
  //     the same loc, while O3 might use a new loc because it temporally overlaps O2, and that new
  //     loc for O3 might be usable for O1, replacing the O1-O2 synthetic dependency with O1-O3,
  //     relaxing the schedule.
  //
  //   * If there are two stresses, O1-O2 and O3-O4, a loc added to relax O1-O2 might also be
  //     usable for relaxing O3-O4.
  //
  //   * Given a stress O1-O2, if there's also a natural dependency between the the steps using
  //     O1 and the step producing O2, there's no actual stress to relieve; in some cases
  //     (depending on cache access patterns) it might even be optimal to continue using the same
  //     loc for O1 and O2.
  //
  //   * Given a stress O1-O2, it's possible that they're different sizes; relieving the stress may
  //     add less than twice the previous memory.

  IVLOG(1, "Final loc count: " << b.locs.size() << " Remaining mem: " << b.mem_available);

  auto result = FinalizeSchedule(&b);
  IVLOG(3, "Final schedule:\n" << result);
  return result;
}

const char* FifoScheduler::name() const { return "FIFO"; }

RunnableStepsIterator& RunnableStepsIterator::operator++() noexcept {
  // Invariant: since we're advancing the iterator, pending_ != nullptr, and
  // pending->at(pos_) is the current value.  Since we're doing a pre-order traversal,
  // the children have not been visited, and the parent has been.

  // Try to visit the left child.
  std::size_t lc = (pos_ << 1) + 1;
  if (lc < pending_->size() && !pending_->at(lc)->dependency_count) {
    pos_ = lc;
    return *this;
  }

  // Try to visit the right child.
  std::size_t rc = lc + 1;
  if (rc < pending_->size() && !pending_->at(rc)->dependency_count) {
    pos_ = rc;
    return *this;
  }

  // Go upwards.  If we're coming from a left child, try to go up and down to the right;
  // otherwise (coming from the right, or unable to go to the left), keep going upwards.
  // Note that left children have odd indicies; right children are even.
  while (pos_) {
    if (pos_ & 1) {
      // This is its parent's left child.
      std::size_t prc = pos_ + 1;
      if (prc < pending_->size() && !pending_->at(prc)->dependency_count) {
        // The parent's right child is a runnable step; visit it.
        pos_ = prc;
        return *this;
      }
    }
    // Either this is its parent's right child, or it's its parent's left child and its
    // parent's right child isn't a runnable step.  Move to the parent and try again.
    pos_ = (pos_ - 1) >> 1;
  }

  // We're at the root; there is no parent, the traversal is done.
  pending_ = nullptr;
  return *this;
}

StepPlan::StepPlan(Build* b, PendingStep* ps) : ps_{ps} {
  // TODO: Consider the order in which we consider the outputs; sorting by size or considering
  // exact-fit-first may produce more-optimal assignments.
  for (const schedule::OutputInfo& oi : ps->step->outputs) {
    bool assigned_output = false;
    std::uint64_t mem_size;
    bool is_io = !oi.allocp->is_tmp();
    if (is_io) {
      mem_size = oi.allocp->byte_size;
    } else {
      mem_size = AlignUp(b, oi.allocp->byte_size);
    }

    auto it = b->value_locs.find(oi.allocp);
    if (it != b->value_locs.end()) {
      // The value is currently in a loc.
      outputs_.emplace_back(it->second);
      continue;
    }
    // Try to find a free loc big enough for this value.
    auto lower_fit = b->free_locs.lower_bound(mem_size);
    auto fit = lower_fit;
    while (fit != b->free_locs.end()) {
      Loc* loc = fit->second;
      if (is_io && loc->byte_size != mem_size) {
        // When assigning IO, we require identical sizes.
        break;
      }
      auto ulres = used_free_locs_.emplace(loc, LocManip{oi.add_dep, is_io, oi.allocp, 0});
      if (!ulres.second) {
        // This free loc's already been used by this plan.
        ++fit;
        continue;
      }
      // We can use this loc.
      free_locs_to_mark_as_used_.emplace_back(fit);
      outputs_.emplace_back(loc);
      assigned_output = true;
      break;
    }
    if (assigned_output) {
      continue;
    }

    // See if there's a free loc we can enlarge to hold this value.
    fit = lower_fit;
    while (fit != b->free_locs.begin()) {
      --fit;
      Loc* loc = fit->second;
      if (loc->is_io) {
        // We can't enlarge IO allocs.
        continue;
      }
      auto ulres = used_free_locs_.emplace(loc, LocManip{oi.add_dep, is_io, oi.allocp, mem_size - loc->byte_size});
      if (!ulres.second) {
        // This free loc's already been used by this plan.
        continue;
      }
      // We can use this loc.
      free_locs_to_mark_as_used_.emplace_back(fit);
      outputs_.emplace_back(loc);
      mem_needed_ += (mem_size - loc->byte_size);
      assigned_output = true;
      break;
    }
    if (assigned_output) {
      continue;
    }

    // We need to create a new loc.
    auto lit = pending_locs_.emplace(pending_locs_.end(), Loc{mem_size});
    lit->add_dep = oi.add_dep;
    lit->contents = oi.allocp;
    outputs_.emplace_back(&*lit);
    mem_needed_ += mem_size;
  }

  // Compute input deltatime sum.
  for (schedule::Alloc* input : ps->step->inputs) {
    input_deltatime_sum_ += b->current_memtime - b->value_locs[input]->cache_memtime;
  }
}

void StepPlan::Apply(Build* b) {
  if (ps_->zero_inputs.size()) {
    std::unordered_map<schedule::Alloc*, Loc*> alloc_locs;
    auto lit = outputs_.begin();
    for (const schedule::OutputInfo& oi : ps_->step->outputs) {
      alloc_locs.emplace(oi.allocp, *lit++);
    }
    for (auto& zero_step : ps_->zero_inputs) {
      auto* zps = zero_step.first;
      auto ss = b->scheduled.emplace(b->scheduled.end(), ScheduledStep{zps->step, zps->work_groups});
      ss->outputs.emplace_back(alloc_locs.at(zero_step.second));
    }
  }
  auto ss = MovePendingStepToScheduled(b, ps_);
  ss->outputs = std::move(outputs_);
  SpliceLocs(b, std::move(pending_locs_), mem_needed_);
  for (auto free_loc_it : free_locs_to_mark_as_used_) {
    MarkLocInUse(b, free_loc_it);
  }
  for (auto& loc_manip : used_free_locs_) {
    Loc* loc = loc_manip.first;
    loc->add_dep |= loc_manip.second.add_dep;
    loc->is_io |= loc_manip.second.is_io;
    loc->contents = loc_manip.second.contents;
    loc->byte_size += loc_manip.second.delta;
  }
}

}  // namespace fifo_scheduler
}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
