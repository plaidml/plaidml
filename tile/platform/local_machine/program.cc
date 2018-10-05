// Copyright 2017-2018 Intel Corporation.

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
#include "tile/platform/local_machine/run_request.h"
#include "tile/proto/support.h"

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

static PerfCounter pre_scan_time("pre_scan_time");
static PerfCounter post_scan_time("post_scan_time");

void AllocateBuffers(const std::vector<std::string>& names, const ShapeMap& types, hal::Memory* memory,
                     std::vector<std::shared_ptr<hal::Buffer>>* buffers) {
  for (const auto& name : names) {
    const auto& shape = types.find(name)->second;
    buffers->push_back(memory->MakeBuffer(shape.byte_size(), hal::BufferAccessMask::ALL));
  }
}

int64_t TryKernel(const context::Context& ctx, const lang::KernelInfo& ki,
                  const std::vector<std::shared_ptr<hal::Buffer>>& buffers, const DevInfo& devinfo, size_t trial_runs) {
  // Check in cache, and early return if found
  int64_t cached_time = lang::TileCache::Instance()->GetDuration(ki.key, ki.settings, ki.tile.shape);
  if (cached_time >= 0) {
    LOG(DEBUG) << "Cached kernel: " << ki.kname << ", key: " << ki.key << ", tile: " << ki.tile.shape;
    return cached_time;
  }

  LOG(DEBUG) << "Trying kernel: " << ki.kname << ", key: " << ki.key << ", tile: " << ki.tile.shape;
  try {
    // Prep to do a real run
    auto& device = *devinfo.dev;
    auto library = device.compiler()->Build(ctx, {ki}, devinfo.settings).get();
    auto executable = device.executor()->Prepare(library.get()).get();
    int64_t best_time = std::numeric_limits<int64_t>::max();

    // Run trial_runs number of times, picking minimum time
    for (size_t i = 0; i < trial_runs; i++) {
      auto evt = executable->Run(ctx, 0, buffers, {}, true);
      device.executor()->Flush();
      auto result = evt->GetFuture().get();
      int64_t time = result->GetDuration().count();
      best_time = std::min(time, best_time);
    }

    // Save in cache and return
    lang::TileCache::Instance()->AddEntry(ki.key, ki.settings, ki.tile.shape, best_time);
    return best_time;
  } catch (const std::exception& ex) {
    LOG(ERROR) << "Skipping kernel failure: " << ex.what();
  } catch (...) {
    LOG(ERROR) << "Skipping unknown kernel failure";
  }
  return std::numeric_limits<int64_t>::max();
}

lang::KernelList CompileProgram(const tile::proto::Program& program, const DevInfo& devinfo,
                                const lang::TileOptimizer& optimizer) {
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
  auto inputs = FromProto(program.inputs());
  auto outputs = FromProto(program.outputs());
  auto settings = hal::settings::ToHardwareSettings(devinfo.settings);
  auto kernel_list = lang::GenerateProgram(parsed, inputs, outputs, settings, optimizer, program.id(), tile_trials);

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
                 const std::shared_ptr<DevInfo>& devinfo, const std::shared_ptr<Scheduler>& scheduler,
                 const std::shared_ptr<MemStrategy>& output_mem_strategy,
                 const std::shared_ptr<MemStrategy>& tmp_mem_strategy, hal::Memory* tmp_memory,
                 const lang::TileOptimizer& optimizer)
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

  kernel_list_ = CompileProgram(program, *devinfo_.get(), optimizer);

  auto lib = devinfo_->dev->compiler()->Build(activity.ctx(), kernel_list_.kernels, devinfo_->settings).get();
  executable_ = devinfo_->dev->executor()->Prepare(lib.get()).get();
  schedule_ = scheduler->BuildSchedule(program, kernel_list_);

  if (activity.ctx().is_logging_events()) {
    hal::proto::CompilationInfo cinfo;
    for (auto kernel : kernel_list_.kernels) {
      (*cinfo.mutable_kernels())[kernel.kname] = kernel.info;
    }
    SummarizeSchedule(&cinfo, program, kernel_list_, schedule_);
    *(cinfo.mutable_program()) = std::move(program);
    activity.AddMetadata(cinfo);
    schedule::proto::Schedule sched_pb;
    schedule::ScheduleToProto(&sched_pb, schedule_);
    for (auto kernel : kernel_list_.kernels) {
      sched_pb.add_knames(kernel.kname);
    }
    activity.AddMetadata(sched_pb);
  }

  ValidateSchedule(program, kernel_list_, schedule_);
}

boost::future<void> Program::Run(const context::Context& ctx,
                                 std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
                                 std::map<std::string, std::shared_ptr<tile::Buffer>> outputs) {
  std::map<std::string, std::shared_ptr<tile::Buffer>> rewrite_outputs;
  for (auto kvp : outputs) {
    rewrite_outputs.emplace(kernel_list_.var_rewrites.Lookup(kvp.first), std::move(kvp.second));
  }
  return RunRequest::Run(ctx, this, std::move(inputs), std::move(rewrite_outputs));
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
