// Copyright 2017, Vertex.AI.

#include "tile/platform/local_machine/scheduler.h"

#include <boost/dynamic_bitset.hpp>

#include <iterator>
#include <map>
#include <set>
#include <unordered_map>

#include "base/util/compat.h"
#include "base/util/error.h"

namespace vertexai {
namespace tile {
namespace local_machine {

namespace {

class InputDepUpdater final : public AllocVisitor {
 public:
  InputDepUpdater(AllocPtr allocp, StepPtr stepp, std::map<AllocPtr, StepPtr, AllocPtrLess>* latest_tmp_writer);

  void Visit(const TmpAlloc& tmp_alloc) final;
  void Visit(const ProgramInputAlloc& input_alloc) final {}
  void Visit(const ProgramOutputAlloc& output_alloc) final;

 private:
  AllocPtr allocp_;
  StepPtr stepp_;
  std::map<AllocPtr, StepPtr, AllocPtrLess>* latest_tmp_writer_;
};

InputDepUpdater::InputDepUpdater(AllocPtr allocp, StepPtr stepp,
                                 std::map<AllocPtr, StepPtr, AllocPtrLess>* latest_tmp_writer)
    : allocp_{allocp}, stepp_{stepp}, latest_tmp_writer_{latest_tmp_writer} {}

void InputDepUpdater::Visit(const TmpAlloc& tmp_alloc) {
  if ((*allocp_)->byte_size) {
    if (!latest_tmp_writer_->count(allocp_)) {
      throw error::Internal{std::string{"Program fails to initialize non-empty temporary for alloc a"} + std::to_string((*allocp_)->idx)};
    }
    IVLOG(5, "  Adding input dep for a" << (*allocp_)->idx << " on last writer s" << (*stepp_)->idx);
    (*stepp_)->deps.insert(latest_tmp_writer_->at(allocp_));
  }
}

void InputDepUpdater::Visit(const ProgramOutputAlloc& out_alloc) {
  IVLOG(5, "  Adding output dep for a" << (*allocp_)->idx << " on last writer s" << (*stepp_)->idx);
  (*stepp_)->deps.insert(latest_tmp_writer_->at(allocp_));
}

class StepDepUpdater final : public StepVisitor {
 public:
  StepDepUpdater(StepPtr stepp, std::map<AllocPtr, StepPtr, AllocPtrLess>* latest_tmp_writer);
  void Visit(const RunStep& run) final;
  void Visit(const CopyStep& copy) final;

 private:
  void AddInput(AllocPtr input);
  void AddOutput(AllocPtr output);

  StepPtr stepp_;
  std::map<AllocPtr, StepPtr, AllocPtrLess>* latest_tmp_writer_;
};

StepDepUpdater::StepDepUpdater(StepPtr stepp, std::map<AllocPtr, StepPtr, AllocPtrLess>* latest_tmp_writer)
    : stepp_{stepp}, latest_tmp_writer_{latest_tmp_writer} {}

void StepDepUpdater::Visit(const RunStep& run) {
  for (auto allocp : run.inputs) {
    AddInput(allocp);
  }
  for (auto oi : run.outputs) {
    AddOutput(oi.allocp);
  }
}

void StepDepUpdater::Visit(const CopyStep& copy) {
  AddInput(copy.from);
  AddOutput(copy.to.allocp);
}

void StepDepUpdater::AddInput(AllocPtr input) {
  InputDepUpdater updater{input, stepp_, latest_tmp_writer_};
  (*input)->Accept(&updater);
}

void StepDepUpdater::AddOutput(AllocPtr output) {
  IVLOG(5, "  Adding output dep for a" << (*output)->idx);
  auto res = latest_tmp_writer_->emplace(std::make_pair(output, stepp_));
  if (res.second) {
    IVLOG(5, "    This was the first writer");
    // This was the first insertion of this alloc; everything is good.
    return;
  }

  IVLOG(5, "    Alloc had been written by s" << (*res.first->second)->idx);

  // The alloc already existed in the map -- so some other step
  // already updated part of the alloc.  The current step has a
  // dependency on that previous writer.
  (*stepp_)->deps.insert(res.first->second);

  // The current step becomes the latest writer.
  res.first->second = stepp_;
}

struct AllocInfo {
  explicit AllocInfo(std::size_t sidx_limit) : accessors(sidx_limit) {}

  boost::dynamic_bitset<> accessors;
  std::string contents;
  bool program_input = false;
  std::size_t last_writer_sidx = 0;
  std::uint64_t byte_size = 0;
};

// Initializes a vector of allocs to the names of the temporary variables they contain at the start of a program, based
// on the program input allocs.
class AllocInit final : private AllocVisitor {
 public:
  static std::vector<AllocInfo> GetAllocContents(const lang::KernelList& kl, const Schedule& schedule) {
    AllocInit init{kl, schedule};
    for (const auto& alloc : schedule.allocs) {
      alloc->Accept(&init);
    }
    return std::move(init.alloc_infos_);
  }

 private:
  explicit AllocInit(const lang::KernelList& kl, const Schedule& schedule)
      : kl_{&kl}, alloc_infos_(schedule.allocs.size(), AllocInfo(schedule.steps.size())) {}

  void Visit(const TmpAlloc& tmp_alloc) final { alloc_infos_[tmp_alloc.idx].byte_size = tmp_alloc.byte_size; }

  void Visit(const ProgramInputAlloc& input_alloc) final {
    AllocInfo& ai = alloc_infos_[input_alloc.idx];
    ai.contents = input_alloc.name;
    ai.program_input = true;
    ai.byte_size = kl_->types.at(input_alloc.name).byte_size();
  }

  void Visit(const ProgramOutputAlloc& output_alloc) final {
    alloc_infos_[output_alloc.idx].byte_size = kl_->types.at(output_alloc.name).byte_size();
  }

  const lang::KernelList* kl_;
  std::vector<AllocInfo> alloc_infos_;
};

// Validates that program outputs end up with the correct tensor values.
class AllocOutputValidator final : private AllocVisitor {
 public:
  static void Validate(const tile::proto::Program& program, const lang::KernelList& kl, const Schedule& schedule,
                       const std::vector<AllocInfo>& alloc_infos) {
    AllocOutputValidator v{program, kl};
    for (const auto& alloc : schedule.allocs) {
      std::size_t aidx = alloc->idx;
      v.contents_ = &alloc_infos[aidx].contents;
      alloc->Accept(&v);
    }
    for (auto& kvp : v.outputs_) {
      if (!kvp.second && kl.types.count(kvp.first)) {
        throw error::Internal{"Schedule fails to write program output \"" + kvp.first + "\""};
      }
    }
  }

 private:
  AllocOutputValidator(const tile::proto::Program& program, const lang::KernelList& kl) {
    for (const auto& output : program.outputs()) {
      outputs_[kl.var_rewrites.Lookup(output.first)] = false;
    }
  }

  void Visit(const TmpAlloc& tmp_alloc) final {}

  void Visit(const ProgramInputAlloc& input_alloc) final {}

  void Visit(const ProgramOutputAlloc& output_alloc) final {
    if (output_alloc.name != *contents_) {
      throw error::Internal{"Schedule ends with tensor \"" + *contents_ + " in program output \"" + output_alloc.name +
                            "\""};
    }
    auto it = outputs_.find(output_alloc.name);
    if (it == outputs_.end()) {
      throw error::Internal{"Schedule writes output \"" + output_alloc.name +
                            "\", which is not defined in the program"};
    }
    if (it->second) {
      throw error::Internal{"Schedule defines program output \"" + output_alloc.name + "\" multiple times"};
    }
    it->second = true;
  }

  std::unordered_map<std::string, bool> outputs_;
  const std::string* contents_ = nullptr;
};

// ScheduleValidator processes the supplied schedule, verifying the invariants specified by ValidateSchedule.
class ScheduleValidator final : private StepVisitor {
 public:
  static void Validate(const tile::proto::Program& program, const lang::KernelList& kl, const Schedule& schedule) {
    ScheduleValidator v(kl, schedule);
    for (const auto& step : schedule.steps) {
      v.CheckDeps(*step);
      step->Accept(&v);
      v.sidx_++;
    }
    if (!v.scheduled_kidxs_.all()) {
      throw error::Internal{"Schedule does not run all kernels"};
    }
    AllocOutputValidator::Validate(program, kl, schedule, v.alloc_infos_);
  }

 private:
  explicit ScheduleValidator(const lang::KernelList& kl, const Schedule& schedule)
      : kl_{&kl},
        scheduled_kidxs_(kl.kernels.size()),
        transitive_deps_(schedule.steps.size(), boost::dynamic_bitset<>(schedule.steps.size())),
        alloc_infos_{AllocInit::GetAllocContents(kl, schedule)} {}

  void Visit(const RunStep& run) final {
    // Verify scheduling.
    if (scheduled_kidxs_.size() <= run.kidx) {
      throw error::Internal{"Schedule contains an invalid kernel index at s" + std::to_string(sidx_)};
    }
    if (scheduled_kidxs_.test(run.kidx)) {
      throw error::Internal{"Schedule runs a kernel multiple times at s" + std::to_string(sidx_)};
    }
    scheduled_kidxs_.set(run.kidx);

    const auto& ki = kl_->kernels[run.kidx];

    // Verify that all inputs are valid.
    if (ki.inputs.size() != run.inputs.size()) {
      throw error::Internal{"Schedule specifies an incorrect number of kernel inputs at s" + std::to_string(sidx_)};
    }
    for (std::size_t iidx = 0; iidx < ki.inputs.size(); ++iidx) {
      std::size_t aidx = (*run.inputs[iidx])->idx;
      CheckAndMarkInput(run.inputs[iidx]);
      if (alloc_infos_[aidx].contents != ki.inputs[iidx]) {
        throw error::Internal{"Schedule specifies an incorrect tensor read at s" + std::to_string(sidx_)};
      }
    }

    // Verify that all outputs are valid.
    if (ki.outputs.size() != run.outputs.size()) {
      throw error::Internal{"Schedule specifies an incorrect number of kernel outputs at s" + std::to_string(sidx_)};
    }
    for (std::size_t oidx = 0; oidx < ki.outputs.size(); ++oidx) {
      CheckAndMarkOutput(run.outputs[oidx].allocp, ki.outputs[oidx]);
    }
  }

  void Visit(const CopyStep& copy) final {
    CheckAndMarkInput(copy.from);
    const std::string& contents = alloc_infos_[(*copy.from)->idx].contents;
    CheckAndMarkOutput(copy.to.allocp, contents);
    std::uint64_t tensor_size = kl_->types.at(contents).byte_size();
    if (tensor_size != copy.byte_count) {
      throw error::Internal{"Schedule copy size mismatch at s" + std::to_string(sidx_) + " (copying " +
                            std::to_string(copy.byte_count) + " bytes; tensor \"" + contents + "\" contains " +
                            std::to_string(tensor_size) + " bytes)"};
    }
  }

  void CheckDeps(const Step& step) {
    for (const auto& dep : step.deps) {
      if (sidx_ <= (*dep)->idx) {
        throw error::Internal{"Schedule deps are not ordered at s" + std::to_string(sidx_)};
      }
      std::size_t dep_sidx = (*dep)->idx;
      transitive_deps_[sidx_].set(dep_sidx);
      transitive_deps_[sidx_] |= transitive_deps_[dep_sidx];
    }
  }

  void CheckAndMarkInput(AllocPtr allocp) {
    std::size_t aidx = (*allocp)->idx;
    if (alloc_infos_.size() <= aidx) {
      throw error::Internal{"Schedule specifies an undefined input alloc at s" + std::to_string(sidx_)};
    }
    auto& ainfo = alloc_infos_[aidx];
    if (!ainfo.contents.length()) {
      throw error::Internal{"Schedule reads an uninitialized tensor at s" + std::to_string(sidx_)};
    }
    if (!ainfo.program_input && !transitive_deps_[sidx_].test(ainfo.last_writer_sidx)) {
      throw error::Internal{"Schedule reads tensor a" + std::to_string(aidx) + " \"" + ainfo.contents + "\" at s" +
                            std::to_string(sidx_) + " prior to its write"};
    }
    ainfo.accessors.set(sidx_);
  }

  void CheckAndMarkOutput(AllocPtr allocp, const std::string& new_contents) {
    std::size_t aidx = (*allocp)->idx;
    if (alloc_infos_.size() <= aidx) {
      throw error::Internal{"Schedule specifies an undefined output alloc at s" + std::to_string(sidx_)};
    }
    auto& ainfo = alloc_infos_[aidx];
    if (ainfo.program_input) {
      throw error::Internal{"Schedule specifies a write to an input tensor at s" + std::to_string(sidx_)};
    }
    // All current accessors should be in the current step's transitive dependency set.
    // Note that we add the current step as a self-dependency in this check, to account for steps
    // that reuse allocs for different temporaries.  TODO: Only add the self-dep for cases where codegen
    // says it's okay to reuse an input alloc for an output.
    auto deps = transitive_deps_[sidx_];
    deps.set(sidx_);
    if (((deps & ainfo.accessors) ^ ainfo.accessors).any()) {
      throw error::Internal{"Schedule writes a tensor to a live alloc at s" + std::to_string(sidx_)};
    }
    std::uint64_t tensor_size = kl_->types.at(new_contents).byte_size();
    if (ainfo.byte_size < tensor_size) {
      throw error::Internal{"Schedule writes tensor \"" + new_contents + "\" of " + std::to_string(tensor_size) +
                            " bytes to alloc a" + std::to_string(aidx) + " of " + std::to_string(ainfo.byte_size) +
                            " bytes at s" + std::to_string(sidx_)};
    }

    // The current step becomes the last writer and the only current
    // accessor.
    ainfo.last_writer_sidx = sidx_;
    ainfo.accessors.reset();
    ainfo.accessors.set(sidx_);
    ainfo.contents = new_contents;
  }

  const lang::KernelList* kl_;
  std::size_t sidx_ = 0;
  boost::dynamic_bitset<> scheduled_kidxs_;
  std::vector<boost::dynamic_bitset<>> transitive_deps_;
  std::vector<AllocInfo> alloc_infos_;
};

}  // namespace

Schedule ToScheduleSteps(const tile::proto::Program& program, const lang::KernelList& kl) {
  Schedule schedule;

  // Figure out the set of program inputs and outputs that will be
  // accessed directly (i.e. without a copy).  These do not require
  // program-local temporaries.
  std::set<std::string> direct_io;
  for (const auto& input : program.inputs()) {
    direct_io.emplace(input.first);
  }
  for (const auto& output : program.outputs()) {
    direct_io.insert(output.first);
  }

  // Create an alloc for every temporary, and remember the tmp->alloc mappings.
  struct TmpInfo {
    AllocPtr allocp;
    bool needs_output_copy = false;
    std::size_t last_output_kidx = 0;
  };

  std::unordered_map<std::string, TmpInfo> tmps;

  for (const auto& ty : kl.types) {
    if (direct_io.count(ty.first)) {
      continue;
    }
    auto alloc = compat::make_unique<TmpAlloc>();
    alloc->byte_size = ty.second.byte_size();
    auto allocp = schedule.allocs.emplace(schedule.allocs.end(), std::move(alloc));
    tmps[ty.first].allocp = allocp;
  }

  // Mark the program inputs.
  for (const auto& input : program.inputs()) {
    auto tyit = kl.types.find(input.first);
    if (tyit == kl.types.end()) {
      // An unused input -- unusual, but possible in some cases.
      continue;
    }
    auto alloc = compat::make_unique<ProgramInputAlloc>();
    alloc->name = input.first;
    alloc->byte_size = tyit->second.byte_size();
    auto allocp = schedule.allocs.emplace(schedule.allocs.end(), std::move(alloc));
    tmps[input.first].allocp = allocp;
  }

  // Mark the program outputs.
  std::set<std::string> program_outputs;
  for (const auto& output_pre_rewrite : program.outputs()) {
    const std::string& output = kl.var_rewrites.Lookup(output_pre_rewrite.first);
    auto tyit = kl.types.find(output);
    if (tyit == kl.types.end()) {
      // An unused output -- unusual, but possible in some cases.
      continue;
    }
    auto alloc = compat::make_unique<ProgramOutputAlloc>();
    alloc->name = output;
    alloc->byte_size = tyit->second.byte_size();
    auto allocp = schedule.allocs.emplace(schedule.allocs.end(), std::move(alloc));
    tmps[output].allocp = allocp;
    program_outputs.insert(output);
  }

  // Since outputs may be generated over multiple kernels, we need to
  // discover the last kernel involved in the construction of any
  // given output.
  for (std::size_t kidx = 0; kidx < kl.kernels.size(); ++kidx) {
    for (const std::string& output : kl.kernels[kidx].outputs) {
      tmps.at(output).last_output_kidx = kidx;
    }
  }

  // Add steps to run the kernels.
  for (std::size_t kidx = 0; kidx < kl.kernels.size(); ++kidx) {
    auto run = compat::make_unique<RunStep>();
    run->kidx = kidx;
    const auto& ki = kl.kernels[kidx];

    // Add the inputs before adding the run step, in case we need to
    // copy inputs locally.
    for (const std::string& input : ki.inputs) {
      run->inputs.emplace_back(tmps.at(input).allocp);
    }

    // Add the outputs to the step.
    for (const std::string& output : ki.outputs) {
      OutputInfo oi{tmps.at(output).allocp, false};
      if (program_outputs.count(output)) {
        oi.add_dep = true;
      }
      auto it = ki.safe_self_aliases.find(output);
      if (it != ki.safe_self_aliases.end()) {
        for (const std::string& safe_self_alias : it->second) {
          const auto& renamed_alias = kl.var_rewrites.Lookup(safe_self_alias);          
          (*oi.allocp)->safe_self_alias_allocs.emplace(tmps.at(renamed_alias).allocp);
        }
      }
      run->outputs.emplace_back(oi);
    }

    // Add the step itself to the schedule.
    schedule.steps.emplace(schedule.steps.end(), std::move(run));
  }

  schedule.Reindex();

  return schedule;
}

void AddDataflowDeps(Schedule* schedule) {
  std::map<AllocPtr, StepPtr, AllocPtrLess> latest_tmp_writer;
  for (auto it = schedule->steps.begin(); it != schedule->steps.end(); ++it) {
    (*it)->deps.clear();
    IVLOG(5, "Adding dataflow deps to s" << (*it)->idx);
    StepDepUpdater updater{it, &latest_tmp_writer};
    (*it)->Accept(&updater);
  }
}

void AddLinearDeps(Schedule* schedule, std::size_t delta) {
  if (schedule->steps.size() <= delta) {
    return;
  }
  auto dep = schedule->steps.begin();
  auto it = dep;
  std::advance(it, delta);
  while (it != schedule->steps.end()) {
    (*it)->deps.insert(dep);
    ++dep;
    ++it;
  }
}

void ValidateSchedule(const tile::proto::Program& program, const lang::KernelList& kl, const Schedule& schedule) {
  ScheduleValidator::Validate(program, kl, schedule);
}

void SummarizeSchedule(hal::proto::CompilationInfo* cinfo, const tile::proto::Program& program,
                       const lang::KernelList& kl, const Schedule& schedule) {
  IVLOG(1, "Schedule for " << program.id() << ":\n" << schedule);

  std::map<std::size_t, std::size_t> tmp_size_counts;
  std::size_t tmp_count = 0;

  for (const auto& kvp : kl.types) {
    if (program.inputs().count(kvp.first) || program.outputs().count(kvp.first)) {
      continue;
    }
    tmp_count++;
    const auto& shape = kvp.second;
    auto res = tmp_size_counts.emplace(shape.byte_size(), 1);
    if (!res.second) {
      res.first->second += 1;
    }
  }

  struct AllocSizeVisitor final : public AllocVisitor {
    void Visit(const TmpAlloc& tmp_alloc) final {
      alloc_count++;
      total_bytes += tmp_alloc.byte_size;
      auto res = size_counts.emplace(tmp_alloc.byte_size, 1);
      if (!res.second) {
        res.first->second += 1;
      }
    }
    void Visit(const ProgramInputAlloc& input_alloc) final {}
    void Visit(const ProgramOutputAlloc& output_alloc) final {}

    std::size_t alloc_count = 0;
    std::uint64_t total_bytes = 0;
    std::map<std::size_t, std::size_t> size_counts;
  };

  AllocSizeVisitor allocs;

  for (const auto& alloc : schedule.allocs) {
    alloc->Accept(&allocs);
  }

  IVLOG(1, "Tmp count: " << tmp_count);
  IVLOG(1, "Tmp sizes: ");
  for (auto it : tmp_size_counts) {
    IVLOG(1, "  " << it.first << ": " << it.second);
    if (cinfo) {
      (*cinfo->mutable_tmp_sizes())[it.first] = it.second;
    }
  }
  IVLOG(1, "Alloc count: " << allocs.alloc_count);
  IVLOG(1, "Alloc sizes: ");
  for (auto it : allocs.size_counts) {
    IVLOG(1, "  " << it.first << ": " << it.second);
    if (cinfo) {
      (*cinfo->mutable_alloc_sizes())[it.first] = it.second;
    }
  }
  IVLOG(1, "Total memory required: " << allocs.total_bytes << " bytes");
}


void ScheduleToProto(proto::Schedule* pb, const Schedule& schedule) {
  
  class AllocToProtoVisitor final : public AllocVisitor {
   public:
    explicit AllocToProtoVisitor(proto::Schedule* pb) : pb_{pb} {}

    void Visit(const TmpAlloc& tmp_alloc) final {
      auto alloc = AddAlloc(tmp_alloc);
      if (tmp_alloc.location == TmpAlloc::ON_DEVICE) {
        alloc->mutable_tmp()->mutable_dev();
      } else {
        alloc->mutable_tmp()->mutable_host();
      }
    }

    void Visit(const ProgramInputAlloc& input_alloc) final {
      auto alloc = AddAlloc(input_alloc);
      alloc->set_input(input_alloc.name);
    }

    void Visit(const ProgramOutputAlloc& output_alloc) final {
      auto alloc = AddAlloc(output_alloc);
      alloc->set_output(output_alloc.name);
    }

  private:
    proto::Alloc* AddAlloc(const Alloc& alloc) {
      auto pb = pb_->add_allocs();
      pb->set_size(alloc.byte_size);
      return pb;
    }

    proto::Schedule* pb_;
  };

  AllocToProtoVisitor alloc_to_proto(pb);

  for (const auto& alloc : schedule.allocs) {
    alloc->Accept(&alloc_to_proto);
  }

  class StepToProtoVisitor final : public StepVisitor {
  public:
    explicit StepToProtoVisitor(proto::Schedule* pb) : pb_{pb} {}

    void Visit(const RunStep& run_step) final {
      auto* run_pb = AddStep(run_step)->mutable_run();
      run_pb->set_kidx(run_step.kidx);
      for (const auto& output : run_step.outputs) {
        run_pb->add_output_aidxs((*output.allocp)->idx);
      }
      for (const auto& input : run_step.inputs) {
        run_pb->add_input_aidxs((*input)->idx);
      }
    }

    void Visit(const CopyStep& copy_step) final {
      auto copy_pb = AddStep(copy_step)->mutable_copy();
      copy_pb->set_from_aidx((*copy_step.from)->idx);
      copy_pb->set_to_aidx((*copy_step.to.allocp)->idx);
      copy_pb->set_count_bytes(copy_step.byte_count);
    }

  private:
    proto::Step* AddStep(const Step& step) {
      auto pb = pb_->add_steps();
      for (auto dep : step.deps) {
        pb->add_deps((*dep)->idx);
      }
      return pb;
    }

    proto::Schedule* pb_;
  };

  StepToProtoVisitor step_to_proto(pb);

  for (const auto& step : schedule.steps) {
    step->Accept(&step_to_proto);
  }
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
