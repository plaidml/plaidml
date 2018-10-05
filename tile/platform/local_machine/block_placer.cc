// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/block_placer.h"

#include <boost/dynamic_bitset.hpp>

#include "base/util/compat.h"
#include "base/util/logging.h"

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

struct TmpInfo;

struct AllocInfo {
  std::unordered_set<TmpInfo*> assigned_tmps;
  std::uint64_t byte_size;
  schedule::Alloc* alloc = nullptr;
  std::string input;
  std::string output;
  bool read_only;
};

struct TmpInfo {
  std::size_t aidx;
  std::size_t sidx_first = 0;
  schedule::Alloc* tmp = nullptr;  // The original alloc associated with this temporary.
  std::uint64_t byte_size;
  std::string input;
  std::string output;
  AllocInfo* assignment = nullptr;
  bool read_only;
};

// Adds a synthetic input-producing step to a schedule.
std::unordered_set<schedule::Alloc*> AddInputStep(const tile::proto::Program& program, schedule::Schedule* schedule) {
  std::vector<schedule::OutputInfo> outputs;
  std::unordered_set<schedule::Alloc*> consumed_inputs;

  schedule::Step step{schedule::Step::Tag::kRun};
  step.idx = 0;
  for (auto& alloc : schedule->allocs) {
    if (!alloc.is_input()) {
      continue;
    }
    if (program.inputs().at(alloc.input).consumed()) {
      step.outputs.push_back(schedule::OutputInfo{&alloc, false});
      consumed_inputs.insert(&alloc);
    }
  }
  schedule->steps.emplace_front(std::move(step));
  return consumed_inputs;
}

// Adds a synthetic output-consuming step to a schedule.
void AddOutputStep(schedule::Schedule* schedule) {
  schedule::Step step{schedule::Step::Tag::kRun};
  step.idx = schedule->steps.back().idx + 1;
  for (auto& alloc : schedule->allocs) {
    if (!alloc.is_output()) {
      continue;
    }
    step.inputs.push_back(&alloc);
  }
  schedule->steps.emplace_back(std::move(step));
}

// Given a schedule, returns a vector of bitsets, one bit and bitset
// for each step.  In each step's bitset, bit N is set iff the step
// has a transitive dependency on step N.
std::vector<boost::dynamic_bitset<>> BuildTransitiveDeps(const schedule::Schedule& schedule) {
  // N.B. We initialize the deps s.t. every step has a dependency on the initial input-producing step.
  std::vector<boost::dynamic_bitset<>> deps(schedule.steps.size(), boost::dynamic_bitset<>(schedule.steps.size(), 1));
  std::size_t sidx = 0;
  for (const auto& step : schedule.steps) {
    for (const auto& dep : step.deps) {
      std::size_t dep_sidx = dep->idx;
      deps[sidx].set(dep_sidx);
      deps[sidx] |= deps[dep_sidx];
    }
    ++sidx;
  }

  // Special case: the final output-consuming step depends on everything.
  deps[schedule.steps.size() - 1].set();

  return deps;
}

// Builds a vector of bitsets, one bitset per alloc, and one bit per
// step.  In each alloc's bitset, bit N is set iff the alloc is
// accessed by step N.
std::vector<boost::dynamic_bitset<>> BuildAllocAccessors(const schedule::Schedule& schedule) {
  std::size_t sidx = 0;
  std::vector<boost::dynamic_bitset<>> accessors{schedule.allocs.size(),
                                                 boost::dynamic_bitset<>(schedule.steps.size())};
  for (const auto& step : schedule.steps) {
    for (schedule::OutputInfo oi : step.outputs) {
      accessors[oi.allocp->idx].set(sidx);
    }
    for (schedule::Alloc* allocp : step.inputs) {
      accessors[allocp->idx].set(sidx);
    }
    sidx++;
  }
  return accessors;
}

// Builds a map of TmpInfo (containing information about the
// schedule's temporary memory allocations), and the schedule's input/output buffer size.
std::map<schedule::Alloc*, TmpInfo> BuildMemInfo(schedule::Schedule* schedule, std::size_t alignment,
                                                 const std::unordered_set<schedule::Alloc*>& consumed_inputs) {
  std::map<schedule::Alloc*, TmpInfo> infos;
  std::size_t aidx = 0;
  for (auto& alloc : schedule->allocs) {
    bool read_only = alloc.is_input() && !consumed_inputs.count(&alloc);
    TmpInfo info;
    info.aidx = aidx;
    info.tmp = &alloc;
    info.byte_size = alloc.byte_size;
    info.input = alloc.input;
    info.output = alloc.output;
    info.read_only = read_only;
    infos.emplace(&alloc, std::move(info));
    ++aidx;
  }
  std::size_t sidx = schedule->steps.size();
  for (auto sit = schedule->steps.rbegin(); sit != schedule->steps.rend(); ++sit) {
    --sidx;
    for (const schedule::OutputInfo& oi : sit->outputs) {
      auto it = infos.find(oi.allocp);
      if (it != infos.end()) {
        it->second.sidx_first = sidx;
      }
    }
  }
  return infos;
}

// Rewrites a schedule's steps according to a set of block placements.
void ApplyStepRewrites(const std::map<schedule::Alloc*, TmpInfo>& tmp_info_map, schedule::Schedule* schedule) {
  auto lookup = [&tmp_info_map](schedule::Alloc* allocp) {
    auto it = tmp_info_map.find(allocp);
    if (it == tmp_info_map.end()) {
      return allocp;
    }
    return it->second.assignment->alloc;
  };

  for (auto& step : schedule->steps) {
    for (auto& oi : step.outputs) {
      oi.allocp = lookup(oi.allocp);
    }
    for (auto& allocp : step.inputs) {
      allocp = lookup(allocp);
    }
  }
}

class BlockPlacement final : public Placement {
 public:
  BlockPlacement(const tile::proto::Program& program, schedule::Schedule* schedule, std::size_t alignment);

  std::uint64_t device_memory_bytes() const final;
  void Apply() final;

 private:
  bool IsCompatible(const std::vector<boost::dynamic_bitset<>>& deps,
                    const std::vector<boost::dynamic_bitset<>>& accessors, const TmpInfo* a, const TmpInfo* b);

  schedule::Schedule* schedule_;
  std::vector<boost::dynamic_bitset<>> tmp_accessors_;
  std::size_t alignment_;
  std::map<schedule::Alloc*, TmpInfo> tmp_info_map_;
  std::vector<AllocInfo> alloc_infos_;
  std::uint64_t sum_ = 0;
};

BlockPlacement::BlockPlacement(const tile::proto::Program& program, schedule::Schedule* schedule, std::size_t alignment)
    : schedule_{schedule}, alignment_{alignment} {
  // Declare routines for creating and augmenting allocs.
  auto create_alloc = [this](TmpInfo* tmp_info) {
    auto ait = alloc_infos_.emplace(alloc_infos_.end(), AllocInfo{});
    ait->byte_size = tmp_info->byte_size;
    ait->assigned_tmps.insert(tmp_info);
    ait->input = tmp_info->input;
    ait->output = tmp_info->output;
    ait->read_only = tmp_info->read_only;
    tmp_info->assignment = &(*ait);
    sum_ += ((tmp_info->byte_size + alignment_ - 1) / alignment_) * alignment_;
  };

  auto add_to_alloc = [](TmpInfo* tmp_info, AllocInfo* alloc_info) {
    alloc_info->assigned_tmps.insert(tmp_info);
    if (!alloc_info->input.length()) {
      alloc_info->input = tmp_info->input;
    }
    if (!alloc_info->output.length()) {
      alloc_info->output = tmp_info->output;
    }
    tmp_info->assignment = alloc_info;
  };

  // In placement:
  //   * The kernel issue ordering is fixed.
  //   * All dependencies are accounted for in the steps.
  //   * All values have been assigned to distinct allocs.
  //
  // It's safe to coalesce two allocs if doing so does not change the
  // values observed by the kernels accessing those allocs.
  //
  // The lifecycle of a value is: the value is written to an alloc,
  // the value is read some number of times, and then the value is
  // destroyed, either at the end of the program or by being
  // overwritten by some other value.  (As supplied, the value is
  // destroyed at the end of the program.)
  //
  // So: A value is written to an alloc (potentially in parts).  Then
  // the alloc is accessed by some number of kernels.  This is the
  // accessor set of the value.
  //
  // When a subsequent value is created and requires an alloc, if the
  // previous value's accessor set is a subset of the dependency set
  // of the subsequent value, then the two values may share an alloc.

  // We start by adding a synthetic output-consuming step to the end
  // of the schedule, allowing output buffers to be used for
  // temporaries while ensuring that they contain their correct final
  // outputs at the end of the program.
  AddOutputStep(schedule);

  // We also add a synthetic input-producing step to the front
  // of the schedule, allowing input buffers to be used for
  // temporaries and outputs while ensuring that they contain their
  // correct input values until fully used by the program.
  std::unordered_set<schedule::Alloc*> consumed_inputs = AddInputStep(program, schedule);
  schedule->Reindex();

  // Next, we build the transitive dependency set of the kernels,
  // and the accessor steps of the various allocs.
  auto deps = BuildTransitiveDeps(*schedule);
  auto accessors = BuildAllocAccessors(*schedule);

  // Next, we extract the existing allocs.
  tmp_info_map_ = BuildMemInfo(schedule_, alignment_, consumed_inputs);
  alloc_infos_.reserve(tmp_info_map_.size());

  // Build a list of the remaining temporaries, and sort it.
  // We want to process inputs and outputs, then non-IO allocs; within each
  // group, we want to process temporaries in largest->smallest order.
  std::list<TmpInfo*> tmp_infos;
  for (auto& kvp : tmp_info_map_) {
    if (kvp.second.assignment) {
      continue;
    }
    if (kvp.second.input.length()) {
      // We handle inputs upfront, since they're guaranteed to not alias.
      create_alloc(&kvp.second);
      continue;
    }
    tmp_infos.emplace_back(&kvp.second);
  }

  tmp_infos.sort([](const TmpInfo* lhs, const TmpInfo* rhs) {
    if (lhs == rhs) {
      return false;
    }
    // N.B. At this point, there will be no inputs.
    auto lo = lhs->output.length();
    auto ro = rhs->output.length();
    if (lo && !ro) {
      return true;
    }
    if (ro && !lo) {
      return false;
    }
    return lhs->byte_size > rhs->byte_size;
  });

  // Create tmp->alloc assignments.  When assigning temporaries, we first try to reuse
  // existing temporary allocs, then try using IO memory, and finally create new
  // allocations when we need one.
  for (TmpInfo* tmp_info : tmp_infos) {
    bool is_output = tmp_info->output.length();
    for (bool consider_io_allocs = is_output;; consider_io_allocs = true) {
      for (auto& alloc_info : alloc_infos_) {
        bool is_io_alloc = alloc_info.input.length() || alloc_info.output.length();
        if ((!consider_io_allocs && is_io_alloc) || (consider_io_allocs && !is_io_alloc)) {
          continue;
        }
        if (is_output && (alloc_info.byte_size != tmp_info->byte_size)) {
          // Require output buffer reuse to be identical size.
          continue;
        }
        if (alloc_info.byte_size < tmp_info->byte_size) {
          continue;
        }
        if (alloc_info.read_only) {
          continue;
        }
        bool compatible = true;
        for (TmpInfo* assigned_tmp : alloc_info.assigned_tmps) {
          if (!IsCompatible(deps, accessors, assigned_tmp, tmp_info)) {
            compatible = false;
            break;
          }
        }
        if (compatible) {
          add_to_alloc(tmp_info, &alloc_info);
          break;
        }
      }
      if (tmp_info->assignment) {
        break;
      }
      if (consider_io_allocs) {
        // We weren't able to find an assignment; we need a new alloc.
        create_alloc(tmp_info);
        break;
      }
    }
  }

  // Remove the synthetic initial and final steps.
  schedule->steps.pop_front();
  schedule->steps.pop_back();
  schedule->Reindex();
}

std::uint64_t BlockPlacement::device_memory_bytes() const { return sum_; }

void BlockPlacement::Apply() {
  // Clear the original allocs.  Note that after this, the original steps and the
  // tmp_info_map_ point to undefined memory; we can continue looking at these
  // as values (for mapping an original alloc to the TmpInfo describing its new
  // assignment), but we must be very careful to never dereference them.
  schedule_->allocs.clear();

  // Create the new allocs.
  for (auto& alloc_info : alloc_infos_) {
    schedule::Alloc alloc;
    alloc.byte_size = alloc_info.byte_size;
    alloc.input = alloc_info.input;
    alloc.output = alloc_info.output;
    alloc_info.alloc = &*schedule_->allocs.emplace(schedule_->allocs.end(), std::move(alloc));
  }

  // Rewrite the existing steps to point to the new allocs.
  ApplyStepRewrites(tmp_info_map_, schedule_);

  schedule_->Reindex();

  IVLOG(1, "Block placer: Schedule uses " << sum_ << " bytes of device memory");
}

bool BlockPlacement::IsCompatible(const std::vector<boost::dynamic_bitset<>>& deps,
                                  const std::vector<boost::dynamic_bitset<>>& accessors, const TmpInfo* a,
                                  const TmpInfo* b) {
  // In order for two temporaries to be compatible, all accessors from
  // one must be deps of the other.
  if (b->sidx_first < a->sidx_first) {
    // Make it so that 'a' is always the earlier-created temporary.
    std::swap(a, b);
  }
  auto intersection = accessors[a->aidx] & deps[b->sidx_first];

  // Now: if b is created by a kernel, and it's written elementwise-safely wrt a by that kernel,
  // we can logically add that kernel to the deps set -- essentially, for any given element in b
  // written by the kernel, there's effectively a dependency on the phase of the kernel that's
  // accessing a.
  if (b->tmp->safe_self_alias_allocs.count(a->tmp)) {
    // N.B. accessors[a->aidx] will already contain b->sidx_first.
    intersection.set(b->sidx_first);
  }

  return intersection == accessors[a->aidx];
}

}  // namespace

BlockPlacer::BlockPlacer(std::size_t alignment) : alignment_{alignment} {}

std::unique_ptr<Placement> BlockPlacer::PlaceSchedule(const tile::proto::Program& program,
                                                      schedule::Schedule* schedule) const {
  return compat::make_unique<BlockPlacement>(program, schedule, alignment_);
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
