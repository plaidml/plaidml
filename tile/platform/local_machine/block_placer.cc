// Copyright 2017, Vertex.AI.

#include "tile/platform/local_machine/block_placer.h"

#include <boost/dynamic_bitset.hpp>

#include "base/util/compat.h"
#include "base/util/logging.h"

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

// Adds a synthetic output-consuming step to a schedule.
class SyntheticOutputConsumer final : private AllocVisitor {
 public:
  static void AddOutputStep(Schedule* schedule) {
    SyntheticOutputConsumer oc;
    for (auto allocp = schedule->allocs.begin(); allocp != schedule->allocs.end(); ++allocp) {
      oc.allocp_ = allocp;
      (*allocp)->Accept(&oc);
    }
    auto step = compat::make_unique<RunStep>();
    step->inputs = std::move(oc.inputs_);
    step->idx = schedule->steps.back()->idx + 1;
    schedule->steps.emplace_back(std::move(step));
  }

 private:
  void Visit(const TmpAlloc& /* tmp_alloc */) final {}
  void Visit(const ProgramInputAlloc& /* input_alloc */) final {}
  void Visit(const ProgramOutputAlloc& /* output_alloc */) final { inputs_.push_back(allocp_); }

  std::vector<AllocPtr> inputs_;
  AllocPtr allocp_;
};

// Given a schedule, returns a vector of bitsets, one bit and bitset
// for each step.  In each step's bitset, bit N is set iff the step
// has a transitive dependency on step N.
std::vector<boost::dynamic_bitset<>> BuildTransitiveDeps(const Schedule& schedule) {
  std::vector<boost::dynamic_bitset<>> deps(schedule.steps.size(), boost::dynamic_bitset<>(schedule.steps.size()));
  std::size_t sidx = 0;
  for (const auto& step : schedule.steps) {
    for (const auto& dep : step->deps) {
      std::size_t dep_sidx = (*dep)->idx;
      deps[sidx].set(dep_sidx);
      deps[sidx] |= deps[dep_sidx];
    }
    ++sidx;
  }
  return deps;
}

// Builds a vector of bitsets, one bitset per alloc, and one bit per
// step.  In each alloc's bitset, bit N is set iff the alloc is
// accessed by step N.
class AllocAccessors final : private StepVisitor {
 public:
  static std::vector<boost::dynamic_bitset<>> Build(const Schedule& schedule) {
    AllocAccessors a{schedule};
    for (const auto& step : schedule.steps) {
      step->Accept(&a);
      ++a.sidx_;
    }
    return std::move(a.accessors_);
  }

 private:
  explicit AllocAccessors(const Schedule& schedule)
      : accessors_(schedule.allocs.size(), boost::dynamic_bitset<>(schedule.steps.size())) {}

  void Visit(const RunStep& run) final {
    for (OutputInfo oi : run.outputs) {
      accessors_[(*oi.allocp)->idx].set(sidx_);
    }
    for (AllocPtr allocp : run.inputs) {
      accessors_[(*allocp)->idx].set(sidx_);
    }
  }

  void Visit(const CopyStep& copy) final {
    accessors_[(*copy.from)->idx].set(sidx_);
    accessors_[(*copy.to.allocp)->idx].set(sidx_);
  }

  std::size_t sidx_ = 0;
  std::vector<boost::dynamic_bitset<>> accessors_;
};

struct TmpInfo;

struct AllocInfo {
  std::set<TmpInfo*> assigned_tmps;
  std::uint64_t byte_size;
  AllocPtr alloc;
  std::string output_name;
};

struct TmpInfo {
  std::size_t aidx;
  std::size_t sidx_first = 0;
  AllocPtr tmp;  // The original alloc associated with this temporary.
  std::uint64_t byte_size;
  std::string output_name;
  AllocInfo* assignment;
};

// Builds a map of TmpInfo (containing information about the
// schedule's temporary memory allocations), and the schedule's input/output buffer size.
class MemInfo final : private AllocVisitor, private StepVisitor {
 public:
  static std::pair<std::map<AllocPtr, TmpInfo, AllocPtrLess>, std::uint64_t> Build(Schedule* schedule,
                                                                                   std::size_t alignment) {
    MemInfo mi{alignment};
    for (mi.current_ = schedule->allocs.begin(); mi.current_ != schedule->allocs.end(); ++mi.current_) {
      (*mi.current_)->Accept(&mi);
      ++mi.aidx_;
    }
    mi.sidx_ = schedule->steps.size();
    for (auto it = schedule->steps.rbegin(); it != schedule->steps.rend(); ++it) {
      --mi.sidx_;
      (*it)->Accept(&mi);
    }
    return std::make_pair(std::move(mi.infos_), mi.io_sum_);
  }

 private:
  explicit MemInfo(std::size_t alignment) : alignment_{alignment} {}

  void Visit(const TmpAlloc& tmp_alloc) final {
    if (tmp_alloc.location != TmpAlloc::ON_DEVICE) {
      return;
    }
    AddTmpInfo(tmp_alloc, "");
  }

  void Visit(const ProgramInputAlloc& input_alloc) final {
    io_sum_ += ((input_alloc.byte_size + alignment_ - 1) / alignment_) * alignment_;
  }

  void Visit(const ProgramOutputAlloc& output_alloc) final {
    io_sum_ += ((output_alloc.byte_size + alignment_ - 1) / alignment_) * alignment_;
    AddTmpInfo(output_alloc, output_alloc.name);
  }

  void AddTmpInfo(const Alloc& alloc, const std::string& output_name) {
    TmpInfo info;
    info.aidx = aidx_;
    info.tmp = current_;
    info.byte_size = alloc.byte_size;
    info.output_name = output_name;
    infos_.emplace(std::make_pair(current_, std::move(info)));
  }

  void Visit(const RunStep& run) final {
    for (OutputInfo oi : run.outputs) {
      SawOutput(oi.allocp);
    }
  }

  void Visit(const CopyStep& copy) final { SawOutput(copy.to.allocp); }

  void SawOutput(AllocPtr output) {
    auto it = infos_.find(output);
    if (it != infos_.end()) {
      it->second.sidx_first = sidx_;
    }
  }

  std::size_t aidx_ = 0;
  std::size_t sidx_;
  AllocPtr current_;
  std::map<AllocPtr, TmpInfo, AllocPtrLess> infos_;
  std::size_t alignment_;
  std::uint64_t io_sum_ = 0;
};

// Rewrites a schedule's steps according to a set of block placements.
class StepRewriter final : private StepVisitor {
 public:
  static void ApplyRewrites(std::map<AllocPtr, TmpInfo, AllocPtrLess>* tmp_info_map, Schedule* schedule) {
    StepRewriter rewriter{tmp_info_map};
    for (const auto& step : schedule->steps) {
      step->Accept(&rewriter);
    }
  }

 private:
  explicit StepRewriter(std::map<AllocPtr, TmpInfo, AllocPtrLess>* tmp_info_map) : tmp_info_map_{tmp_info_map} {}

  void Visit(const RunStep& const_run) final {
    RunStep* run = const_cast<RunStep*>(&const_run);
    for (auto& oi : run->outputs) {
      oi.allocp = Lookup(oi.allocp);
    }
    for (auto& allocp : run->inputs) {
      allocp = Lookup(allocp);
    }
  }

  void Visit(const CopyStep& const_copy) final {
    CopyStep* copy = const_cast<CopyStep*>(&const_copy);
    copy->to.allocp = Lookup(copy->to.allocp);
    copy->from = Lookup(copy->from);
  }

 private:
  AllocPtr Lookup(AllocPtr p) {
    auto it = tmp_info_map_->find(p);
    if (it == tmp_info_map_->end()) {
      return p;
    }
    return it->second.assignment->alloc;
  }

  std::map<AllocPtr, TmpInfo, AllocPtrLess>* tmp_info_map_;
};

class BlockPlacement final : public Placement {
 public:
  BlockPlacement(Schedule* schedule, std::size_t alignment);

  std::uint64_t device_memory_bytes() const final;
  void Apply() final;

 private:
  bool IsCompatible(const std::vector<boost::dynamic_bitset<>>& deps,
                    const std::vector<boost::dynamic_bitset<>>& accessors, const TmpInfo* a, const TmpInfo* b);

  Schedule* schedule_;
  std::vector<boost::dynamic_bitset<>> tmp_accessors_;
  std::size_t alignment_;
  std::map<AllocPtr, TmpInfo, AllocPtrLess> tmp_info_map_;
  std::vector<AllocInfo> alloc_infos_;
  std::uint64_t sum_ = 0;
};

BlockPlacement::BlockPlacement(Schedule* schedule, std::size_t alignment) : schedule_{schedule}, alignment_{alignment} {
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
  SyntheticOutputConsumer::AddOutputStep(schedule);

  // Next, we build the transitive dependency set of the kernels,
  // and the accessor steps of the various allocs.
  auto deps = BuildTransitiveDeps(*schedule_);
  auto accessors = AllocAccessors::Build(*schedule_);

  // Next, we extract the existing temporary allocs, and sort them by
  // size, biggest-first.
  std::tie(tmp_info_map_, sum_) = MemInfo::Build(schedule_, alignment_);
  std::vector<TmpInfo*> tmp_infos;
  tmp_infos.reserve(tmp_info_map_.size());
  for (auto& kvp : tmp_info_map_) {
    tmp_infos.emplace_back(&kvp.second);
  }
  std::sort(tmp_infos.begin(), tmp_infos.end(),
            [](const TmpInfo* lhs, const TmpInfo* rhs) { return lhs->byte_size > rhs->byte_size; });

  // Pre-populate the allocations with program output buffers.
  alloc_infos_.reserve(tmp_infos.size());

  for (TmpInfo* tmp_info : tmp_infos) {
    if (!tmp_info->output_name.size()) {
      continue;
    }
    auto ait = alloc_infos_.emplace(alloc_infos_.end(), AllocInfo{});
    ait->byte_size = tmp_info->byte_size;
    ait->assigned_tmps.insert(tmp_info);
    ait->output_name = tmp_info->output_name;
    tmp_info->assignment = &(*ait);
  }

  // Create tmp->alloc assignments, largest->smallest.
  for (TmpInfo* tmp_info : tmp_infos) {
    if (tmp_info->output_name.size()) {
      continue;
    }
    bool assigned = false;
    for (auto& alloc_info : alloc_infos_) {
      if (alloc_info.byte_size < tmp_info->byte_size) {
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
        alloc_info.assigned_tmps.insert(tmp_info);
        tmp_info->assignment = &alloc_info;
        assigned = true;
        break;
      }
    }
    if (!assigned) {
      // We need a new alloc for this temporary.
      auto ait = alloc_infos_.emplace(alloc_infos_.end(), AllocInfo{});
      ait->byte_size = tmp_info->byte_size;
      ait->assigned_tmps.insert(tmp_info);
      tmp_info->assignment = &(*ait);
      sum_ += ((tmp_info->byte_size + alignment_ - 1) / alignment_) * alignment_;
    }
  }

  // Remove the synthetic final step.
  schedule->steps.pop_back();
}

std::uint64_t BlockPlacement::device_memory_bytes() const { return sum_; }

void BlockPlacement::Apply() {
  // Create the new allocs.
  for (auto& alloc_info : alloc_infos_) {
    if (alloc_info.output_name.length()) {
      auto alloc = compat::make_unique<ProgramOutputAlloc>();
      alloc->name = alloc_info.output_name;
      alloc->byte_size = alloc_info.byte_size;
      alloc_info.alloc = schedule_->allocs.emplace(schedule_->allocs.end(), std::move(alloc));
    } else {
      auto alloc = compat::make_unique<TmpAlloc>();
      alloc->byte_size = alloc_info.byte_size;
      alloc_info.alloc = schedule_->allocs.emplace(schedule_->allocs.end(), std::move(alloc));
    }
  }

  // Rewrite the existing steps to point to the new allocs.
  StepRewriter::ApplyRewrites(&tmp_info_map_, schedule_);

  // Erase the original allocs.
  for (const auto& kvp : tmp_info_map_) {
    schedule_->allocs.erase(kvp.first);
  }

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
  if ((*b->tmp)->safe_self_alias_allocs.count(a->tmp)) {
    // N.B. accessors[a->aidx] will already contain b->sidx_first.
    intersection.set(b->sidx_first);
  }

  return intersection == accessors[a->aidx];
}

}  // namespace

BlockPlacer::BlockPlacer(std::size_t alignment) : alignment_{alignment} {}

std::unique_ptr<Placement> BlockPlacer::PlaceSchedule(Schedule* schedule) const {
  return compat::make_unique<BlockPlacement>(schedule, alignment_);
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
