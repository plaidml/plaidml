// Copyright 2017, Vertex.AI.

#include "tile/platform/local_machine/program.h"

#include <algorithm>
#include <forward_list>
#include <numeric>
#include <set>
#include <unordered_set>
#include <utility>

#include "base/util/error.h"
#include "base/util/perf_counter.h"
#include "tile/hal/util/settings.h"
#include "tile/lang/parser.h"
#include "tile/lang/tile_cache.h"
#include "tile/platform/local_machine/buffer.h"
#include "tile/proto/support.h"

namespace vertexai {
namespace tile {
namespace local_machine {

namespace {

static PerfCounter pre_scan_time("pre_scan_time");
static PerfCounter post_scan_time("post_scan_time");

void AllocateBuffers(const std::vector<std::string>& names, const lang::ShapeMap& types, hal::Memory* memory,
                     std::vector<std::shared_ptr<hal::Buffer>>* buffers) {
  for (const auto& name : names) {
    const auto& shape = types.find(name)->second;
    buffers->push_back(memory->MakeBuffer(shape.byte_size(), hal::BufferAccessMask::ALL));
  }
}

int64_t TryKernel(const context::Context& ctx, const lang::KernelInfo& ki,
                  const std::vector<std::shared_ptr<hal::Buffer>>& buffers, const DevInfo& devinfo, size_t trial_runs) {
  // Check in cache, and early return if found
  int64_t cached_time = lang::TileCache::Instance()->GetDuration(ki.key, ki.settings, ki.tile_size);
  if (cached_time >= 0) {
    LOG(DEBUG) << "Cached kernel: " << ki.key;
    return cached_time;
  }

  // Prep to do a real run
  auto& device = *devinfo.dev;
  auto library = device.compiler()->Build(ctx, {ki}, devinfo.settings).get();
  auto kernel = device.executor()->Prepare(library.get(), 0).get();
  int64_t best_time = std::numeric_limits<int64_t>::max();

  // Run trial_runs number of times, picking minimum time
  for (size_t i = 0; i < trial_runs; i++) {
    LOG(DEBUG) << "Trying kernel: " << ki.kname << ", key: " << ki.key << ", tile_size: " << ki.tile_size;
    auto evt = kernel->Run(ctx, buffers, {});
    device.executor()->Flush();
    auto result = evt->GetFuture().get();
    int64_t time = result->GetDuration().count();
    best_time = std::min(time, best_time);
  }

  // Save in cache and return
  lang::TileCache::Instance()->AddEntry(ki.key, ki.settings, ki.tile_size, best_time);
  return best_time;
}

lang::KernelList CompileProgram(const tile::proto::Program& program, const DevInfo& devinfo) {
  IVLOG(2, "Compiling: " << program.code());
  size_t tile_trials = 1;
  size_t trial_runs = 1;
  if (program.has_tile_scanning_params()) {
    tile_trials = program.tile_scanning_params().max_trials();
    trial_runs = program.tile_scanning_params().max_trial_runs();
  }

  context::Context ctx;
  lang::Parser parser;
  auto parsed = parser.Parse(program.code());
  auto inputs = to_poco(program.inputs());
  auto outputs = to_poco(program.outputs());
  auto settings = hal::settings::ToHardwareSettings(devinfo.settings);
  auto kernel_list = lang::GenerateProgram(parsed, inputs, outputs, settings, program.id(), tile_trials);

  if (tile_trials == 1) {
    return kernel_list;
  }

  auto& device = *devinfo.dev;
  auto memory = device.executor()->shared_memory();
  if (!memory) {
    memory = device.executor()->device_memory();
    if (!memory) {
      memory = devinfo.devset->host_memory();
      if (!memory) {
        throw std::runtime_error("Device memory not available");
      }
    }
  }

  for (auto& ki : kernel_list.kernels) {
    if (ki.candidates.empty()) {
      continue;
    }

    std::vector<std::shared_ptr<hal::Buffer>> buffers;
    AllocateBuffers(ki.outputs, kernel_list.types, memory, &buffers);
    AllocateBuffers(ki.inputs, kernel_list.types, memory, &buffers);

    std::vector<lang::KernelInfo> candidates;
    std::swap(candidates, ki.candidates);

    size_t cur_num = 0;
    size_t best_num = 0;
    uint64_t best_time = TryKernel(ctx, ki, buffers, devinfo, trial_runs);
    pre_scan_time.add(best_time);
    for (const auto& candidate : candidates) {
      cur_num++;
      uint64_t time = TryKernel(ctx, candidate, buffers, devinfo, trial_runs);
      if (time < best_time) {
        best_time = time;
        ki = candidate;
        best_num = cur_num;
      }
    }
    post_scan_time.add(best_time);
    IVLOG(1, "  best: " << double(best_time) / 1e9 << ", index: " << best_num);
    IVLOG(1, "  pre_scan_time: " << double(pre_scan_time.get()) / 1e9
                                 << ", post_scan_time: " << double(post_scan_time.get()) / 1e9);
  }

  return kernel_list;
}

}  // namespace

Program::Program(const context::Context& ctx, const tile::proto::Program& program,
                 const std::shared_ptr<DevInfo>& devinfo, const std::shared_ptr<MemStrategy>& output_mem_strategy,
                 const std::shared_ptr<MemStrategy>& tmp_mem_strategy, hal::Memory* tmp_memory)
    : devinfo_{devinfo}, output_mem_strategy_{output_mem_strategy}, tmp_mem_strategy_{tmp_mem_strategy} {
  // TODO: Make this path asynchronous.
  // Asynchronous programming is a little tricky in this case, since if we compile asynchronously, the
  // compilation may not be complete when we're first asked to run a program, which means we'd need to save the run
  // request and mark the output buffers somehow to indicate that they're not just not ready due to hal events, they're
  // not ready due to compilation futures (which can't be waited on simultaneously).  There's a relatively
  // straightforward dataflow-ish way to describe the resulting system, but it's more complicated than just
  // compiling everything synchronously, so we just do everything synchronously for now.

  if (!devinfo->dev->compiler() || !devinfo->dev->executor()) {
    // TODO: Implement a mechanism for providing a pre-compiled program.
    throw error::Unavailable{"The requested device is unavailable for running Tile programs"};
  }

  context::Activity activity{ctx, "tile::local_machine::Compile"};

  hal::proto::CompilationInfo cinfo;

  lang::KernelList kernel_list = CompileProgram(program, *devinfo_.get());
  for (auto kernel : kernel_list.kernels) {
    (*cinfo.mutable_kernels())[kernel.kname] = kernel.info;
  }

  var_rewrites_ = std::move(kernel_list.var_rewrites);

  LoadKernels(activity.ctx(), std::move(kernel_list.kernels));
  PushOutputConsumer(program);
  auto tmps = AllocTemporaries(program, kernel_list.types);
  AddInterKernelDeps(16);
  ScheduleTemporaries(std::move(tmps));
  PopOutputConsumer();
  LogTemporaries(&cinfo);
  ValidateTemporaries();

  *(cinfo.mutable_program()) = std::move(program);
  activity.AddMetadata(cinfo);
}

void Program::LoadKernels(const context::Context& ctx, std::vector<lang::KernelInfo> kernel_infos) {
  auto lib = devinfo_->dev->compiler()->Build(ctx, kernel_infos, devinfo_->settings).get();

  kernels_.reserve(kernel_infos.size() + 1);
  for (std::size_t kidx = 0; kidx < kernel_infos.size(); ++kidx) {
    BoundKernel bk;
    bk.info = std::move(kernel_infos[kidx]);
    // Prepare the kernel for execution (loading it into the GPU, making it executable, &c).
    bk.kernel = devinfo_->dev->executor()->Prepare(lib.get(), kidx).get();
    kernels_.emplace_back(std::move(bk));
  }
}

void Program::PushOutputConsumer(const tile::proto::Program& program) {
  // We push an extra nop kernel onto the end of the bound kernels
  // list, with an input dependency on every program output.  To the
  // scheduler logic, this looks like something requiring every
  // program output to be live simultaneously, preventing it from
  // reusing output buffers for intra-program temporaries once those
  // buffers' final outputs have been written.
  BoundKernel bk;
  for (const auto& output : program.outputs()) {
    bk.info.inputs.emplace_back(var_rewrites_.Lookup(output.first));
  }
  bk.info.kname = "<synthetic output collector>";
  kernels_.emplace_back(std::move(bk));
}

void Program::PopOutputConsumer() { kernels_.pop_back(); }

std::vector<Program::TmpInfo> Program::AllocTemporaries(const tile::proto::Program& program,
                                                        const lang::ShapeMap& shape_map) {
  // The map of temporary buffer names->indicies.
  std::map<std::string, std::size_t> tmp_name_to_tidx;

  // Information about each temporary, by index.
  std::vector<TmpInfo> tmps;

  // A helper for initializing temporaries.
  auto get_tidx = [&](std::size_t kidx, const std::string& bname) {
    auto tiv = tmp_name_to_tidx.insert(std::make_pair(bname, 0));
    if (tiv.second) {  // true iff inserted into the name->tidx table
      auto tidx = tmp_name_to_tidx.size() - 1;
      IVLOG(4, "Inserted temp " << bname << ", allocating tidx=" << tidx);
      tmps.resize(tidx + 1);
      tiv.first->second = tidx;

      // Compute the temporary's size.
      const auto& ty = shape_map.at(bname);
      std::uint64_t elem_size = ty.elem_size();
      std::uint64_t byte_size = ty.byte_size();

      IVLOG(4, "  Temp " << bname << " tidx=" << tidx << " elem_size=" << elem_size << " byte_size=" << byte_size);

      // Save the computed size and creator.
      tmps[tidx].first_writer_kidx = kidx;
      tmps[tidx].elem_size = elem_size;
      tmps[tidx].byte_size = byte_size;
    } else {
      IVLOG(4, "Temp " << bname << " already had tidx=" << tiv.first->second);
    }
    return tiv.first->second;
  };

  // Translate the program output names, so that we can look up kernel output names in it.
  std::unordered_set<std::string> rewrite_program_outputs;
  for (const auto& kvp : program.outputs()) {
    rewrite_program_outputs.insert(var_rewrites_.Lookup(kvp.first));
  }

  for (std::size_t kidx_current = 0; kidx_current < kernels_.size(); ++kidx_current) {
    BoundKernel& bk = kernels_[kidx_current];
    IVLOG(4, "Binding parameters for kidx=" << kidx_current << ": " << to_string(bk.info));

    // Set up output parameters (N.B. outputs come before inputs).
    for (auto bname : bk.info.outputs) {
      if (rewrite_program_outputs.count(bname)) {
        auto tidx = get_tidx(kidx_current, bname);
        bk.params.push_back(KernelParam{KernelParamType::kOutput, bname, tidx});
        tmps[tidx].program_output = bname;
        continue;
      }

      bk.params.push_back(KernelParam{KernelParamType::kTmpOutput, bname, get_tidx(kidx_current, bname)});
    }

    // Set up input parameters.
    for (auto bname : bk.info.inputs) {
      auto it = program.inputs().find(bname);
      if (it != program.inputs().end()) {
        bk.params.push_back(KernelParam{KernelParamType::kInput, bname});
        continue;
      }

      // A kernel input might also be a program output produced by an earlier kernel.
      if (rewrite_program_outputs.count(bname)) {
        bk.params.push_back(KernelParam{KernelParamType::kTmpInput, bname, get_tidx(kidx_current, bname)});
        continue;
      }

      auto tidx = get_tidx(kidx_current, bname);
      bool war_safe_reader = false;
      if (bk.info.war_safe_reads.count(bname)) {
        war_safe_reader = true;
      }
      bk.params.push_back(KernelParam{KernelParamType::kTmpInput, bname, tidx, war_safe_reader});
    }
  }

  return tmps;
}

void Program::AddInterKernelDeps(size_t max_in_flight) {
  IVLOG(4, "Adding inter-kernel synthetic dependencies");
  for (std::size_t kidx = max_in_flight; kidx < kernels_.size(); ++kidx) {
    kernels_[kidx].dep_kidxs.insert(kidx - max_in_flight);
  }
}

void Program::ScheduleTemporaries(std::vector<TmpInfo> tmps) {
  // Assumptions:
  //   * The kernel issue ordering is fixed
  //   * All dependencies are accounted for in the kernel input
  //     parameters.
  // This is a bit simplistic; eventually, we should reorder kernels
  // and add synthetic dependencies and buffer swap operations, so
  // that we can bound the device memory used by a program at runtime.

  // First, we construct the kernel transitive dependency sets: for
  // each kernel, the set of all kernels that are known to have
  // completed when the kernel is run.
  //
  // Along the way, we also construct the set of kernels that access
  // each temporary.
  std::vector<std::set<std::size_t>> kernel_deps;
  kernel_deps.resize(kernels_.size());

  std::vector<std::set<std::size_t>> tmp_accessors;
  tmp_accessors.resize(tmps.size());

  std::vector<std::set<std::size_t>> war_safe_readers;
  war_safe_readers.resize(tmps.size());

  for (std::size_t kidx = 0; kidx < kernels_.size(); ++kidx) {
    auto& bk = kernels_[kidx];
    auto dest = std::inserter(kernel_deps[kidx], kernel_deps[kidx].end());
    for (const auto& param : bk.params) {
      if (param.ty != KernelParamType::kTmpInput && param.ty != KernelParamType::kTmpOutput) {
        continue;
      }
      tmp_accessors[param.tidx].insert(kidx);
      if (tmps[param.tidx].first_writer_kidx == kidx) {
        // This is where the temporary is created.  The current kernel
        // is not considered to be in its own transitive dependency
        // set.
        tmps[param.tidx].last_writer_kidx = kidx;
        continue;
      }
      if (param.war_safe_reader) {
        war_safe_readers[param.tidx].insert(kidx);
      }
      std::size_t last_writer_kidx = tmps[param.tidx].last_writer_kidx;
      dest = last_writer_kidx;
      bk.dep_kidxs.insert(last_writer_kidx);
      const auto& writer_deps = kernel_deps[last_writer_kidx];
      std::copy(writer_deps.begin(), writer_deps.end(), dest);
      if (param.ty == KernelParamType::kTmpOutput) {
        tmps[param.tidx].last_writer_kidx = kidx;
      }
    }
  }

  IVLOG(4, "Built tmp accessor sets");

  // Using this data, we can tell if two temporaries are temporally
  // distinct: they're distinct if all accessors of one temporary are
  // in the set of indirect dependencies of the creator of the other
  // temporary (and thus in the indirect dependencies of every
  // accessor of the other temporary).
  //
  // One special case: if the creator of the second temporary is also
  // an accessor of the first, and the creator of the second temporary
  // guarantees that all writes to the second temporary's buffer
  // strictly follow all reads from the same memory locations of the
  // first temporary's buffer (which is a property of the way the
  // kernel accesses the first temporary), the creator doesn't count
  // as an accessor of the first temporary -- if this is the only
  // dependency keeping them from being temporally distinct, they're
  // temporally distinct.
  auto is_distinct = [&](std::size_t tidx_a, std::size_t tidx_b) {
    if (tmps[tidx_b].first_writer_kidx < tmps[tidx_a].first_writer_kidx) {
      std::swap(tidx_a, tidx_b);
    }
    const auto& b_deps = kernel_deps[tmps[tidx_b].first_writer_kidx];
    for (auto kidx_a : tmp_accessors[tidx_a]) {
      if (!b_deps.count(kidx_a)) {
        return false;
      }
    }
    return true;
  };

  // Allocate alloc tracking structures, and initialize tmp_locs_
  std::vector<std::set<std::size_t>> alloc_tmps;
  alloc_tmps.reserve(tmps.size());
  tmp_locs_.resize(tmps.size());

  // Sort the temporaries, from largest to smallest.  Along the way,
  // we'll filter out the temporaries that are actually program
  // outputs, creating named allocs for them (since their assignments
  // can't be changed).
  std::vector<std::size_t> tidx_by_size;
  tidx_by_size.reserve(tmps.size());
  for (std::size_t tidx = 0; tidx < tmps.size(); ++tidx) {
    if (tmps[tidx].program_output.length()) {
      auto aidx = alloc_tmps.size();
      IVLOG(4, "  Allocating output aidx=" << aidx << " for tidx=" << tidx
                                           << " program_output=" << tmps[tidx].program_output);
      alloc_tmps.resize(aidx + 1);
      alloc_tmps[aidx].insert(tidx);
      alloc_infos_.resize(aidx + 1);
      alloc_infos_[aidx].byte_size = tmps[tidx].byte_size;
      alloc_infos_[aidx].program_output = tmps[tidx].program_output;
      tmp_locs_[tidx] = aidx;
    } else {
      tidx_by_size.push_back(tidx);
    }
  }
  std::sort(tidx_by_size.begin(), tidx_by_size.end(), [&](std::size_t lhs_tidx, std::size_t rhs_tidx) {
    return tmps[lhs_tidx].byte_size > tmps[rhs_tidx].byte_size;
  });

  IVLOG(4, "Built tidx_by_size vector: " << tidx_by_size);

  // For each temporary, we assign the first existing alloc whose
  // existing temporaries are guaranteed to not temporally overlap the
  // temporary under consideration.  If we can't find a workable
  // alloc, we create one.
  //
  // It does matter which alloc we choose -- if there are two allocs
  // available, a subsequent temporary that temporally overlaps the
  // current temporary might only not-overlap with a subset of the
  // allocs currently available for the current temporary.  For now,
  // we're arbitrarily picking the first matching alloc, but
  // long-term, it might be interesting to do a complete search.

  for (std::size_t tidx : tidx_by_size) {
    IVLOG(4, "Considering tidx=" << tidx);
    bool used_existing_alloc = false;
    for (std::size_t aidx = 0; aidx < alloc_tmps.size(); ++aidx) {
      IVLOG(4, "  Considering alloc aidx=" << aidx);
      if (alloc_infos_[aidx].byte_size < tmps[tidx].byte_size) {
        IVLOG(4, "    Alloc is too small");
        continue;
      }
      auto& alloc = alloc_tmps[aidx];
      bool alloc_usable = true;
      for (std::size_t alloc_tidx : alloc) {
        IVLOG(4, "    Considering existing tidx=" << alloc_tidx);
        if (!is_distinct(tidx, alloc_tidx)) {
          IVLOG(4, "      tidx=" << tidx << " conflicts with tidx=" << alloc_tidx);
          alloc_usable = false;
          break;
        }
      }
      if (alloc_usable) {
        IVLOG(4, "    Using alloc aidx=" << aidx << " for tidx=" << tidx);
        alloc.insert(tidx);
        tmp_locs_[tidx] = aidx;
        used_existing_alloc = true;
        break;
      }
    }
    if (!used_existing_alloc) {
      auto aidx = alloc_tmps.size();
      IVLOG(4, "  Failed to find an existing alloc; allocating aidx=" << aidx << " for tidx=" << tidx);
      alloc_tmps.resize(aidx + 1);
      alloc_tmps[aidx].insert(tidx);
      alloc_infos_.resize(aidx + 1);
      alloc_infos_[aidx].byte_size = tmps[tidx].byte_size;
      tmp_locs_[tidx] = aidx;
    }
  }

  IVLOG(4, "Done assigning allocs");
}

void Program::LogTemporaries(hal::proto::CompilationInfo* cinfo) {
  std::map<std::size_t, std::size_t> tmp_size_counts;
  std::map<std::size_t, std::size_t> alloc_size_counts;
  for (const auto& tl : tmp_locs_) {
    if (alloc_infos_[tl].program_output.length()) {
      continue;
    }
    auto res = tmp_size_counts.emplace(alloc_infos_[tl].byte_size, 1);
    if (!res.second) {
      res.first->second += 1;
    }
  }

  std::size_t total_alloced = 0;
  std::size_t total_allocs = 0;

  for (const auto& al : alloc_infos_) {
    if (al.program_output.length()) {
      continue;
    }
    total_allocs++;
    total_alloced += al.byte_size;
    auto res = alloc_size_counts.emplace(al.byte_size, 1);
    if (!res.second) {
      res.first->second += 1;
    }
  }

  IVLOG(4, "Tmp count: " << tmp_locs_.size());
  IVLOG(4, "Tmp sizes: ");
  for (auto it : tmp_size_counts) {
    IVLOG(4, "  " << it.first << ": " << it.second);
    (*cinfo->mutable_tmp_sizes())[it.first] = it.second;
  }
  IVLOG(4, "Alloc count: " << total_allocs);
  IVLOG(4, "Alloc sizes: ");
  for (auto it : alloc_size_counts) {
    IVLOG(4, "  " << it.first << ": " << it.second);
    (*cinfo->mutable_alloc_sizes())[it.first] = it.second;
  }
  IVLOG(4, "Total memory allocated: " << total_alloced);
}

void Program::ValidateTemporaries() {
  // Given that the temporary buffer locations have been determined,
  // we want to validate that the resulting allocations are correct.
  //
  // Invariant: if two temporary buffers overlap, their lifetimes must
  // not overlap: the sets of kernels that access each must not
  // overlap, and every kernel that accesses the temporary with the
  // later lifetime must have a dependency (possibly indirect) on
  // every kernel that accesses the temporary with the earlier
  // lifetime.
  //
  // Since earlier temporaries are allocated before later temporaries,
  // the earlier temporary's lifetime should be the earlier of the two
  // -- that is, every one of the later temporary's kernels should
  // recursively depend on every one of the earlier temporary's
  // kernels.
  //
  // We start by scanning the kernels.  As we go, we construct two
  // vectors:
  //
  // * A mapping from each temporary to the set of kernels that access
  //   the temporary (with a flag to indicate whether the accessor is
  //   a war-safe reader).
  //
  // * A mapping from each kernel to the dependencies of that kernel
  //   (aka the "known to have finished" set), which we build by
  //   knowing the temporaries that the kernel accesses, the last
  //   writer of each temporary, and the set of dependencies for each
  //   of those writers.
  std::vector<std::map<std::size_t, bool>> tidx_to_accessor_kidxs(tmp_locs_.size());
  std::vector<std::set<std::size_t>> kidx_to_dep_kidxs(kernels_.size());

  IVLOG(4, "  Building dependency lists for placement verification");

  {
    for (std::size_t kidx_current = 0; kidx_current < kernels_.size(); ++kidx_current) {
      IVLOG(4, "    Considering kidx=" << kidx_current);
      const auto& bk = kernels_[kidx_current];
      for (auto dep_kidx : bk.dep_kidxs) {
        const std::set<std::size_t>& dep_kidxs = kidx_to_dep_kidxs[dep_kidx];
        IVLOG(4, "      Adding direct dep kidx=" << dep_kidx);
        kidx_to_dep_kidxs[kidx_current].insert(dep_kidx);
        IVLOG(4, "      Adding transitive deps kidx=" << dep_kidxs);
        kidx_to_dep_kidxs[kidx_current].insert(dep_kidxs.begin(), dep_kidxs.end());
      }
      for (const auto& param : bk.params) {
        if (param.ty == KernelParamType::kTmpInput || param.ty == KernelParamType::kTmpOutput) {
          IVLOG(4, "      Considering param tidx=" << param.tidx);
          tidx_to_accessor_kidxs[param.tidx].emplace(kidx_current, param.war_safe_reader);
        }
      }
    }
  }

  // Then, we order the temporaries by alloc.
  struct IdxTmpLoc {
    std::size_t tidx;
    std::uint64_t aidx;
  };

  std::vector<IdxTmpLoc> locs;
  locs.reserve(tmp_locs_.size());
  for (std::size_t tidx = 0; tidx < tmp_locs_.size(); ++tidx) {
    locs.emplace_back(IdxTmpLoc{tidx, tmp_locs_[tidx]});
  }

  std::sort(locs.begin(), locs.end(), [](const IdxTmpLoc& lhs, const IdxTmpLoc& rhs) { return lhs.aidx < rhs.aidx; });

  // Next, scan the temporaries, and make sure that there's a
  // dependency path between all of the pairs of kernels of each
  // spatial overlap.
  for (auto it = locs.begin(); it != locs.end(); ++it) {
    for (auto check_it = it + 1; check_it != locs.end(); ++check_it) {
      if (it->aidx != check_it->aidx) {
        // We've moved on to the next aidx.
        break;
      }

      IVLOG(4, "  Spatial overlap: tidx=" << it->tidx << " aidx=" << it->aidx << " and tidx=" << check_it->tidx
                                          << " aidx=" << check_it->aidx);

      std::size_t tidx_low, tidx_high;
      if (it->tidx < check_it->tidx) {
        tidx_low = it->tidx;
        tidx_high = check_it->tidx;
      } else {
        tidx_low = check_it->tidx;
        tidx_high = it->tidx;
      }

      for (auto kidx_war_high : tidx_to_accessor_kidxs[tidx_high]) {
        auto kidx_high = kidx_war_high.first;
        const auto& deps = kidx_to_dep_kidxs[kidx_high];
        for (auto kidx_war_low : tidx_to_accessor_kidxs[tidx_low]) {
          auto kidx_low = kidx_war_low.first;
          if (!deps.count(kidx_low) && (kidx_high != kidx_low || !kidx_war_low.second)) {
            // N.B. If we're using the write-after-read check, we might
            // want to validate that the temporaries have the same
            // buffer and element sizes.  We currently don't, because by
            // the time this code is run, we've dropped the temporary
            // size information, and we're careful to validate this in
            // the call to is_distinct().
            LOG(FATAL) << "Internal logic error: kidx=" << kidx_high << " accesses tidx=" << tidx_high
                       << " aidx=" << tmp_locs_[tidx_high] << " while kidx=" << kidx_low
                       << " accesses tidx=" << tidx_low << " aidx=" << tmp_locs_[tidx_low];
          }
        }
      }
    }
  }
}

namespace {

// Represents the state of a Program::Run request.
class RunRequest {
 public:
  static boost::future<void> BuildAndIssue(const context::Context& ctx, Program* program,
                                           std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
                                           std::map<std::string, std::shared_ptr<tile::Buffer>> outputs);

 private:
  struct PendingUpdate {
    std::shared_ptr<tile::Buffer> input_to_reassign;
    std::shared_ptr<MemChunk> new_input_value;
  };

  struct KernelLogInfo {
    std::shared_ptr<hal::Event> done;
    std::string kname;
    std::size_t tot_bytes;
    std::size_t tot_flops;
  };

  RunRequest(Program* program, std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
             std::map<std::string, std::shared_ptr<tile::Buffer>> outputs);
  void Log();
  void AllocTemporaries(const context::Context& ctx);
  void BuildChunkMap(const context::Context& ctx);
  std::forward_list<PendingUpdate> DealiasIO(const context::Context& ctx);
  boost::future<std::vector<std::shared_ptr<hal::Result>>> LaunchKernels(const context::Context& ctx);
  void ApplyIOUpdates(std::forward_list<PendingUpdate> io_updates);
  boost::future<void> LogResults(const context::Context& ctx, boost::future<std::vector<std::shared_ptr<hal::Result>>>);

  Program* program_;
  std::vector<std::shared_ptr<MemChunk>> tmps_;
  std::map<std::string, std::shared_ptr<tile::Buffer>> inputs_;
  std::map<std::string, std::shared_ptr<tile::Buffer>> outputs_;
  std::map<std::string, std::shared_ptr<MemChunk>> input_chunk_map_;
  std::map<std::string, std::shared_ptr<MemChunk>> output_chunk_map_;
};

boost::future<void> RunRequest::BuildAndIssue(const context::Context& ctx, Program* program,
                                              std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
                                              std::map<std::string, std::shared_ptr<tile::Buffer>> outputs) {
  RunRequest req{program, std::move(inputs), std::move(outputs)};
  req.Log();

  context::Activity running{ctx, "tile::local_machine::Program::Run"};
  boost::future<void> complete;

  {
    context::Activity queueing{running.ctx(), "tile::local_machine::Program::Enqueue"};

    req.BuildChunkMap(queueing.ctx());
    auto io_updates = req.DealiasIO(queueing.ctx());
    req.AllocTemporaries(queueing.ctx());

    auto results = req.LaunchKernels(queueing.ctx());

    req.ApplyIOUpdates(std::move(io_updates));

    complete = req.LogResults(queueing.ctx(), std::move(results));
  }

  // Keep the request and activity referenced until the program is complete.
  return complete.then([ req = std::move(req), running = std::move(running) ](decltype(complete) fut) { fut.get(); });
}

RunRequest::RunRequest(Program* program, std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
                       std::map<std::string, std::shared_ptr<tile::Buffer>> outputs)
    : program_{program}, inputs_{std::move(inputs)}, outputs_{std::move(outputs)} {}

void RunRequest::Log() {
  if (VLOG_IS_ON(4)) {
    VLOG(4) << "Running program " << program_;
    for (const auto& it : inputs_) {
      VLOG(4) << "Input  " << it.first << " -> Buffer " << it.second.get() << " -> HAL Buffer "
              << Buffer::Downcast(it.second, program_->devinfo())->chunk()->hal_buffer().get();
    }
    for (const auto& it : outputs_) {
      VLOG(4) << "Output " << it.first << " -> Buffer " << it.second.get() << " -> HAL Buffer "
              << Buffer::Downcast(it.second, program_->devinfo())->chunk()->hal_buffer().get();
    }
    for (std::size_t kidx = 0; kidx < program_->kernels().size(); ++kidx) {
      const auto& bk = program_->kernels()[kidx];
      VLOG(4) << "Kernel " << bk.info.kname << " kidx=" << kidx << ":";
      for (const auto& param : bk.params) {
        switch (param.ty) {
          case Program::KernelParamType::kInput:
            VLOG(4) << " <= " << param.name;
            break;
          case Program::KernelParamType::kOutput:
            VLOG(4) << " => " << param.name;
            break;
          case Program::KernelParamType::kTmpInput:
            VLOG(4) << " <- " << param.name << " tidx=" << param.tidx;
            break;
          case Program::KernelParamType::kTmpOutput:
            VLOG(4) << " -> " << param.name << " tidx=" << param.tidx;
            break;
        }
      }
    }
    for (std::size_t tidx = 0; tidx < program_->tmp_locs().size(); ++tidx) {
      const auto& aidx = program_->tmp_locs()[tidx];
      const auto& info = program_->alloc_infos()[aidx];
      VLOG(4) << "tidx=" << tidx << ": aidx=" << aidx << " size=" << info.byte_size << " output=\""
              << info.program_output << "\"";
    }
  }
}

void RunRequest::AllocTemporaries(const context::Context& ctx) {
  if (program_->tmp_locs().size()) {
    std::vector<std::shared_ptr<MemChunk>> allocs;
    allocs.reserve(program_->alloc_infos().size());
    for (const auto& info : program_->alloc_infos()) {
      if (info.program_output.length() == 0) {
        allocs.emplace_back(program_->tmp_mem_strategy()->MakeChunk(ctx, info.byte_size));
      } else {
        allocs.emplace_back(output_chunk_map_[info.program_output]);
      }
    }

    tmps_.reserve(program_->tmp_locs().size());
    for (const auto& alloc_idx : program_->tmp_locs()) {
      tmps_.emplace_back(allocs[alloc_idx]);
    }
  }
}

boost::future<std::vector<std::shared_ptr<hal::Result>>> RunRequest::LaunchKernels(const context::Context& ctx) {
  std::vector<std::shared_ptr<hal::Event>> kernel_events;
  kernel_events.resize(program_->kernels().size());

  try {
    for (std::size_t kidx = 0; kidx < program_->kernels().size(); ++kidx) {
      const auto& bk = program_->kernels()[kidx];

      IVLOG(1, "Launching tile program: " << to_string(bk.info));

      std::shared_ptr<hal::Event> done;

      std::vector<std::shared_ptr<hal::Buffer>> params;
      params.reserve(bk.params.size());

      std::vector<std::shared_ptr<hal::Event>> deps;
      deps.reserve(bk.dep_kidxs.size());
      for (auto dep_kidx : bk.dep_kidxs) {
        assert(dep_kidx < kidx);
        deps.emplace_back(kernel_events[dep_kidx]);
      }

      // Set up the output buffers, and add them to the kernel
      // dependencies in case some earlier kernel (like a zeroing
      // kernel) is writing to the same buffer.
      for (const auto& param : bk.params) {
        switch (param.ty) {
          case Program::KernelParamType::kInput: {
            auto it = input_chunk_map_.find(param.name);
            if (it == input_chunk_map_.end()) {
              // It's possible for a program output to be used as a
              // subsequent kernel input; if the kernel input isn't
              // found in the input chunk map, it'll be in the output
              // chunk map.
              it = output_chunk_map_.find(param.name);
              if (it == output_chunk_map_.end()) {
                throw error::Internal{std::string{"Unable to look up kernel input \""} + param.name +  // NOLINT
                                      "\" at runtime"};
              }
            }
            auto in_chunk = it->second;
            IVLOG(2, "  Input: " << in_chunk->size() << " bytes in HAL Buffer " << in_chunk->hal_buffer().get());
            params.emplace_back(in_chunk->hal_buffer());
            auto indeps = in_chunk->deps()->GetReadDependencies();
            for (auto& dep : indeps) {
              IVLOG(2, "    Adding ordinary input dep");
              deps.emplace_back(std::move(dep));
            }
            break;
          }

          case Program::KernelParamType::kOutput: {
            auto out_chunk = output_chunk_map_.at(param.name);
            IVLOG(2, "  Output: " << out_chunk->size() << " bytes in HAL Buffer " << out_chunk->hal_buffer().get());
            params.emplace_back(out_chunk->hal_buffer());
            auto outdeps = out_chunk->deps()->GetReadDependencies();
            for (auto& dep : outdeps) {
              IVLOG(2, "    Adding ordinary output dep");
              deps.emplace_back(std::move(dep));
            }
            break;
          }

          case Program::KernelParamType::kTmpInput:
            IVLOG(2, "  TmpInput tidx=" << param.tidx);
            params.emplace_back(tmps_[param.tidx]->hal_buffer());
            break;

          case Program::KernelParamType::kTmpOutput:
            IVLOG(2, "  TmpOutput tidx=" << param.tidx);
            params.emplace_back(tmps_[param.tidx]->hal_buffer());
            break;
        }
      }

      done = bk.kernel->Run(ctx, params, deps);

      for (const auto& param : bk.params) {
        if (param.ty != Program::KernelParamType::kOutput) {
          continue;
        }
        output_chunk_map_.at(param.name)->deps()->AddReadDependency(done);
      }

      kernel_events[kidx] = done;
    }
  } catch (...) {
    // Any error in the launch poisons all output buffers.
    for (auto& op : output_chunk_map_) {
      op.second->deps()->Poison(std::current_exception());
    }
  }

  return program_->devinfo()->dev->executor()->WaitFor(kernel_events);
}

void RunRequest::BuildChunkMap(const context::Context& ctx) {
  // Prepare the buffers for attaching to kernels
  for (const auto& kvp : inputs_) {
    input_chunk_map_.emplace(kvp.first, Buffer::Downcast(kvp.second, program_->devinfo())->chunk());
  }
  for (const auto& kvp : outputs_) {
    output_chunk_map_.emplace(kvp.first, Buffer::Downcast(kvp.second, program_->devinfo())->chunk());
  }
}

std::forward_list<RunRequest::PendingUpdate> RunRequest::DealiasIO(const context::Context& ctx) {
  // The list of input buffers to be updated to output buffers after all kernels have been issued.
  std::forward_list<PendingUpdate> io_updates;

  // Build a set of all input buffers.
  std::unordered_set<std::shared_ptr<tile::Buffer>> input_buffers;
  for (auto in : inputs_) {
    input_buffers.emplace(in.second);
  }

  // Examine kernels; de-alias if an input is used after an output is produced in the same buffer.
  for (std::size_t kidx = 0; kidx < program_->kernels().size(); ++kidx) {
    VLOG(4) << "Checking for aliasing for kernel " << kidx;
    const auto& bk = program_->kernels()[kidx];
    for (const auto& param : bk.params) {
      if (param.ty != Program::KernelParamType::kOutput) {
        continue;
      }
      VLOG(4) << "  Checking for aliasing for output " << param.name;
      auto oit = outputs_.find(param.name);
      if (oit == outputs_.end()) {
        // This output isn't a program output; nothing to do.
        VLOG(4) << "    " << param.name << " is not a program output";
        continue;
      }
      auto iit = input_buffers.find(oit->second);
      if (iit == input_buffers.end()) {
        // This output buffer wasn't used as an input; nothing to do.
        VLOG(4) << "    " << param.name << " is not used as an input";
        continue;
      }

      // TODO: Add the operation dependency graph to the program,
      // making it available for computing dealiasing, so that we
      // don't have to de-duplicate every time a buffer is used as
      // both an input and an output.

      // This output aliases an input; use a new chunk for the output.
      auto buf = std::make_shared<Buffer>(program_->devinfo(),
                                          program_->output_mem_strategy()->MakeChunk(ctx, oit->second->size()));
      IVLOG(4, "    " << param.name << " aliases an input; replacing with Buffer " << buf.get() << " -> HAL Buffer "
                      << buf->chunk()->hal_buffer().get());
      output_chunk_map_[param.name] = buf->chunk();

      // Reassign the output, so that any further uses of output also reveal the new buffer.
      // Not that this should happen in a correct Tile program, but it's safe to be paranoid --
      // and this also gives us a single location to find the chunks to poison if kernel launch fails.
      oit->second = buf;

      // Remember to update the input later.
      io_updates.emplace_front(PendingUpdate{*iit, buf->chunk()});
    }
  }

  return io_updates;
}

void RunRequest::ApplyIOUpdates(std::forward_list<PendingUpdate> io_updates) {
  for (auto update : io_updates) {
    Buffer::Downcast(update.input_to_reassign, program_->devinfo())->RemapTo(update.new_input_value);
  }
}

boost::future<void> RunRequest::LogResults(const context::Context& ctx,
                                           boost::future<std::vector<std::shared_ptr<hal::Result>>> results) {
  context::Context ctx_copy{ctx};
  return results.then([ctx = std::move(ctx_copy)](decltype(results) future) {
    auto results = future.get();
    if (VLOG_IS_ON(1)) {
      std::chrono::high_resolution_clock::duration total{std::chrono::high_resolution_clock::duration::zero()};
      for (const auto& result : results) {
        total += result->GetDuration();
        result->LogStatistics();
      }
      VLOG(1) << "Total program execution duration: " << total.count();
    }
  });
}

}  // namespace

boost::future<void> Program::Run(const context::Context& ctx,
                                 std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
                                 std::map<std::string, std::shared_ptr<tile::Buffer>> outputs) {
  for (const auto& it : outputs) {
    VLOG(4) << "Original output " << it.first << " -> Buffer " << it.second.get() << " -> HAL Buffer "
            << Buffer::Downcast(it.second, devinfo())->chunk()->hal_buffer().get();
  }
  std::map<std::string, std::shared_ptr<tile::Buffer>> rewrite_outputs;
  for (auto kvp : outputs) {
    rewrite_outputs.emplace(var_rewrites_.Lookup(kvp.first), std::move(kvp.second));
  }
  return RunRequest::BuildAndIssue(ctx, this, std::move(inputs), std::move(rewrite_outputs));
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
