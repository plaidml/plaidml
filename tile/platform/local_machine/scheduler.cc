// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/scheduler.h"

#include <iterator>
#include <map>
#include <set>
#include <unordered_map>

#include <boost/dynamic_bitset.hpp>

#include "base/util/compat.h"
#include "base/util/error.h"

namespace vertexai {
namespace tile {
namespace local_machine {

namespace {

struct AllocInfo {
  explicit AllocInfo(std::size_t sidx_limit) : accessors(sidx_limit) {}

  boost::dynamic_bitset<> accessors;
  std::string contents;
  bool program_input = false;
  bool read_only = false;
  std::size_t last_writer_sidx = 0;
  std::uint64_t byte_size = 0;
};

// Returns a vector of alloc infos with the names of the temporary variables they contain at the start of a program,
// based
// on the program input allocs.
std::vector<AllocInfo> GetAllocContents(const tile::proto::Program& program, const schedule::Schedule& schedule) {
  std::vector<AllocInfo> alloc_infos{schedule.allocs.size(), AllocInfo{schedule.steps.size()}};
  for (const auto& alloc : schedule.allocs) {
    AllocInfo& ai = alloc_infos[alloc.idx];
    ai.byte_size = alloc.byte_size;
    if (alloc.is_input()) {
      ai.contents = alloc.input;
      ai.program_input = true;
      ai.read_only = !program.inputs().at(alloc.input).consumed();
    }
  }
  return alloc_infos;
}

}  // namespace

schedule::Schedule ToScheduleSteps(const tile::proto::Program& program, const lang::KernelList& kl) {
  schedule::Schedule schedule;

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
    schedule::Alloc* allocp;
    bool needs_output_copy = false;
    std::size_t last_output_kidx = 0;
  };

  std::unordered_map<std::string, TmpInfo> tmps;

  for (const auto& ty : kl.types) {
    if (direct_io.count(ty.first)) {
      continue;
    }
    schedule::Alloc alloc;
    alloc.byte_size = ty.second.byte_size();
    auto ait = schedule.allocs.emplace(schedule.allocs.end(), std::move(alloc));
    tmps[ty.first].allocp = &*ait;
  }

  // Mark the program inputs.
  for (const auto& input : program.inputs()) {
    auto tyit = kl.types.find(input.first);
    if (tyit == kl.types.end()) {
      // An unused input -- unusual, but possible in some cases.
      continue;
    }
    schedule::Alloc alloc;
    alloc.input = input.first;
    alloc.byte_size = tyit->second.byte_size();
    auto ait = schedule.allocs.emplace(schedule.allocs.end(), std::move(alloc));
    tmps[input.first].allocp = &*ait;
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
    schedule::Alloc alloc;
    alloc.output = output;
    alloc.byte_size = tyit->second.byte_size();
    auto ait = schedule.allocs.emplace(schedule.allocs.end(), std::move(alloc));
    tmps[output].allocp = &*ait;
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
    schedule::Step run{schedule::Step::Tag::kRun};
    run.kidx = kidx;
    const auto& ki = kl.kernels[kidx];

    // Add the inputs before adding the run step, in case we need to
    // copy inputs locally.
    for (const std::string& input : ki.inputs) {
      run.inputs.emplace_back(tmps.at(input).allocp);
    }

    // Add the outputs to the step.
    for (const std::string& output : ki.outputs) {
      schedule::OutputInfo oi{tmps.at(output).allocp, false};
      if (program_outputs.count(output)) {
        oi.add_dep = true;
      }
      auto it = ki.safe_self_aliases.find(output);
      if (it != ki.safe_self_aliases.end()) {
        for (const std::string& safe_self_alias : it->second) {
          const auto& renamed_alias = kl.var_rewrites.Lookup(safe_self_alias);
          oi.allocp->safe_self_alias_allocs.emplace(tmps.at(renamed_alias).allocp);
        }
      }
      run.outputs.emplace_back(oi);
    }

    // Add the step itself to the schedule.
    schedule.steps.emplace(schedule.steps.end(), std::move(run));
  }

  schedule.Reindex();

  return schedule;
}

void AddDataflowDeps(schedule::Schedule* schedule) {
  std::map<schedule::Alloc*, schedule::Step*> latest_tmp_writer;
  for (auto& step : schedule->steps) {
    step.deps.clear();
    IVLOG(5, "Adding dataflow deps to s" << step.idx);
    for (schedule::Alloc* allocp : step.inputs) {
      if (!allocp->byte_size) {
        return;
      }
      auto ltwit = latest_tmp_writer.find(allocp);
      if (ltwit == latest_tmp_writer.end()) {
        if (!allocp->is_input()) {
          std::stringstream ss;
          ss << "Program fails to initialize non-empty temporary for a" << allocp->idx;
          throw error::Internal(ss.str());
        }
      } else {
        IVLOG(5, "  Adding dep on a" << allocp->idx << " with last writer s" << ltwit->second->idx);
        step.deps.insert(ltwit->second);
      }
    }
    for (schedule::OutputInfo oi : step.outputs) {
      schedule::Alloc* allocp = oi.allocp;
      IVLOG(5, "  Adding output dep on a" << allocp->idx);
      auto res = latest_tmp_writer.emplace(allocp, &step);
      if (res.second) {
        IVLOG(5, "    This was the first writer");
        // This was the first insertion of this alloc; everything is good.
        continue;
      }

      IVLOG(5, "    Alloc had been written by s" << res.first->second->idx);

      // The alloc already existed in the map -- so some other step
      // already updated part of the alloc.  The current step has a
      // dependency on that previous writer.
      step.deps.insert(res.first->second);

      // The current step becomes the latest writer.
      res.first->second = &step;
    }
  }
}

void AddLinearDeps(schedule::Schedule* schedule, std::size_t delta) {
  if (schedule->steps.size() <= delta) {
    return;
  }
  auto dep = schedule->steps.begin();
  auto it = dep;
  std::advance(it, delta);
  while (it != schedule->steps.end()) {
    it->deps.insert(&*dep);
    ++dep;
    ++it;
  }
}

void ValidateSchedule(const tile::proto::Program& program, const lang::KernelList& kl,
                      const schedule::Schedule& schedule) {
  boost::dynamic_bitset<> scheduled_kidxs{kl.kernels.size()};
  std::vector<boost::dynamic_bitset<>> transitive_deps{schedule.steps.size(),
                                                       boost::dynamic_bitset<>(schedule.steps.size())};
  std::vector<AllocInfo> alloc_infos = GetAllocContents(program, schedule);

  std::size_t sidx = 0;
  for (const auto& step : schedule.steps) {
    for (const auto& dep : step.deps) {
      if (sidx <= dep->idx) {
        throw error::Internal{"Schedule deps are not ordered at s" + std::to_string(sidx)};
      }
      std::size_t dep_sidx = dep->idx;
      transitive_deps[sidx].set(dep_sidx);
      transitive_deps[sidx] |= transitive_deps[dep_sidx];
    }

    std::vector<std::string> new_contents;

    switch (step.tag) {
      case schedule::Step::Tag::kRun: {
        // Verify scheduling.
        if (scheduled_kidxs.size() <= step.kidx) {
          throw error::Internal{"Schedule contains an invalid kernel index at s" + std::to_string(sidx)};
        }
        if (scheduled_kidxs.test(step.kidx)) {
          throw error::Internal{"Schedule runs a kernel multiple times at s" + std::to_string(sidx)};
        }
        scheduled_kidxs.set(step.kidx);
        const auto& ki = kl.kernels[step.kidx];
        // Verify that all inputs are valid.
        if (ki.inputs.size() != step.inputs.size()) {
          throw error::Internal{"Schedule specifies an incorrect number of kernel inputs at s" + std::to_string(sidx)};
        }
        for (std::size_t iidx = 0; iidx < ki.inputs.size(); ++iidx) {
          std::size_t aidx = step.inputs[iidx]->idx;
          if (alloc_infos[aidx].contents != ki.inputs[iidx]) {
            throw error::Internal{"Schedule specifies an incorrect tensor read at s" + std::to_string(sidx)};
          }
        }
        // Verify that all outputs are valid.
        if (ki.outputs.size() != step.outputs.size()) {
          throw error::Internal{"Schedule specifies an incorrect number of kernel outputs at s" + std::to_string(sidx)};
        }
        new_contents = ki.outputs;
        break;
      }
      case schedule::Step::Tag::kCopy: {
        if (step.outputs.size() != 1) {
          throw error::Internal{
              "In schedule validation: copy step " + std::to_string(sidx) +
              " has an incorrect output count; expected: 1, got: " + std::to_string(step.outputs.size())};
        }
        if (step.inputs.size() != 1) {
          throw error::Internal{
              "In schedule validation: copy step " + std::to_string(sidx) +
              " has an incorrect input count; expected: 1, got: " + std::to_string(step.inputs.size())};
        }
        const std::string& contents = alloc_infos[step.inputs[0]->idx].contents;
        std::uint64_t tensor_size = kl.types.at(contents).byte_size();
        if (tensor_size != step.byte_count) {
          throw error::Internal{"Schedule copy size mismatch at s" + std::to_string(sidx) + " (copying " +
                                std::to_string(step.byte_count) + " bytes; tensor \"" + contents + "\" contains " +
                                std::to_string(tensor_size) + " bytes)"};
        }
        new_contents.emplace_back(contents);
        break;
      }
      default:
        throw error::Internal{"In validation: step " + std::to_string(sidx) + " has an invalid tag"};
    }

    for (auto* allocp : step.inputs) {
      std::size_t aidx = allocp->idx;
      if (alloc_infos.size() <= aidx) {
        throw error::Internal{"Schedule specifies an undefined input alloc at s" + std::to_string(sidx)};
      }
      auto& ainfo = alloc_infos[aidx];
      if (!ainfo.contents.length()) {
        throw error::Internal{"Schedule reads an uninitialized tensor at s" + std::to_string(sidx)};
      }
      if (!ainfo.program_input && !transitive_deps[sidx].test(ainfo.last_writer_sidx)) {
        throw error::Internal{"Schedule reads tensor a" + std::to_string(aidx) + " \"" + ainfo.contents + "\" at s" +
                              std::to_string(sidx) + " prior to its write"};
      }
      ainfo.accessors.set(sidx);
    }

    for (std::size_t oidx = 0; oidx < new_contents.size(); ++oidx) {
      auto* allocp = step.outputs[oidx].allocp;

      std::size_t aidx = allocp->idx;
      if (alloc_infos.size() <= aidx) {
        throw error::Internal{"Schedule specifies an undefined output alloc at s" + std::to_string(sidx)};
      }
      auto& ainfo = alloc_infos[aidx];
      if (ainfo.read_only) {
        LOG(ERROR) << "Schedule specified a write to a read-only tensor: " << step;
        LOG(ERROR) << schedule;
        throw error::Internal{"Schedule specifies a write to a read-only tensor at s" + std::to_string(sidx)};
      }
      // All current accessors should be in the current step's transitive dependency set.
      // Note that we add the current step as a self-dependency in this check, to account for steps
      // that reuse allocs for different temporaries.  TODO: Only add the self-dep for cases where codegen
      // says it's okay to reuse an input alloc for an output.
      auto deps = transitive_deps[sidx];
      deps.set(sidx);
      if (((deps & ainfo.accessors) ^ ainfo.accessors).any()) {
        throw error::Internal{"Schedule writes a tensor to a live alloc at s" + std::to_string(sidx)};
      }
      std::uint64_t tensor_size = kl.types.at(new_contents[oidx]).byte_size();
      if (ainfo.byte_size < tensor_size) {
        throw error::Internal{"Schedule writes tensor \"" + new_contents[oidx] + "\" of " +
                              std::to_string(tensor_size) + " bytes to alloc a" + std::to_string(aidx) + " of " +
                              std::to_string(ainfo.byte_size) + " bytes at s" + std::to_string(sidx)};
      }

      // The current step becomes the last writer and the only current
      // accessor.
      ainfo.last_writer_sidx = sidx;
      ainfo.accessors.reset();
      ainfo.accessors.set(sidx);
      ainfo.contents = new_contents[oidx];
    }

    sidx++;
  }

  if (!scheduled_kidxs.all()) {
    throw error::Internal{"Schedule does not run all kernels"};
  }

  std::unordered_map<std::string, bool> outputs;
  for (const auto& output : program.outputs()) {
    outputs[kl.var_rewrites.Lookup(output.first)] = false;
  }

  for (const auto& alloc : schedule.allocs) {
    if (!alloc.is_output()) {
      continue;
    }
    const std::string& contents = alloc_infos[alloc.idx].contents;
    if (alloc.output != contents) {
      throw error::Internal{"Schedule ends with tensor \"" + contents + " in program output \"" + alloc.output + "\""};
    }
    if (alloc.byte_size != alloc_infos[alloc.idx].byte_size) {
      throw error::Internal{"Schedule writes output " + alloc.output + " [size=" + std::to_string(alloc.byte_size) +
                            " to an alloc size=" + std::to_string(alloc_infos[alloc.idx].byte_size)};
    }
    auto it = outputs.find(alloc.output);
    if (it == outputs.end()) {
      throw error::Internal{"Schedule writes output \"" + alloc.output + "\", which is not defined in the program"};
    }
    if (it->second) {
      throw error::Internal{"Schedule defines program output \"" + alloc.output + "\" multiple times"};
    }
    it->second = true;
  }
  for (auto& kvp : outputs) {
    if (!kvp.second && kl.types.count(kvp.first)) {
      throw error::Internal{"Schedule fails to write program output \"" + kvp.first + "\""};
    }
  }
}

void SummarizeSchedule(hal::proto::CompilationInfo* cinfo, const tile::proto::Program& program,
                       const lang::KernelList& kl, const schedule::Schedule& schedule) {
  IVLOG(1, "Summary for " << program.id() << ":");

  IVLOG(3, "Schedule:\n" << schedule);

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

  std::size_t alloc_count = 0;
  std::uint64_t total_bytes = 0;
  std::map<std::size_t, std::size_t> size_counts;

  for (const auto& alloc : schedule.allocs) {
    alloc_count++;
    total_bytes += alloc.byte_size;
    auto res = size_counts.emplace(alloc.byte_size, 1);
    if (!res.second) {
      res.first->second += 1;
    }
  }

  IVLOG(2, "Tmp count: " << tmp_count);
  IVLOG(2, "Tmp sizes: ");
  for (auto it : tmp_size_counts) {
    IVLOG(2, "  " << it.first << ": " << it.second);
    if (cinfo) {
      (*cinfo->mutable_tmp_sizes())[it.first] = it.second;
    }
  }
  IVLOG(1, "Alloc count: " << alloc_count);
  IVLOG(1, "Alloc sizes: ");
  for (auto it : size_counts) {
    IVLOG(1, "  " << it.first << ": " << it.second);
    if (cinfo) {
      (*cinfo->mutable_alloc_sizes())[it.first] = it.second;
    }
  }
  IVLOG(1, "Total memory required: " << total_bytes << " bytes");
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
