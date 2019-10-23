// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/program.h"

#include <algorithm>
#include <forward_list>
#include <limits>
#include <numeric>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "base/util/env.h"
#include "base/util/error.h"
#include "base/util/perf_counter.h"
#include "tile/hal/util/settings.h"
#include "tile/lang/parser.h"
#include "tile/lang/tile_cache.h"
#include "tile/ocl_exec/stripe_gen.h"
#include "tile/platform/local_machine/buffer.h"
#include "tile/platform/local_machine/run_request.h"
#include "tile/proto/support.h"

namespace vertexai {
namespace tile {
namespace local_machine {

namespace {

// For Program synchonization
std::size_t avail_mem = 0;
std::once_flag first_run;
std::condition_variable cond_var;
std::mutex mutex;

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

lang::KernelList CompileProgram(           //
    const tile::proto::Program& program,   //
    const DevInfo& devinfo,                //
    const lang::TileOptimizer& optimizer,  //
    ConstBufferManager* const_bufs) {
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

  auto use_stripe_default = devinfo.settings.use_stripe() ? "1" : "0";
  auto use_stripe = env::Get("PLAIDML_USE_STRIPE", use_stripe_default) == "1";
  if (use_stripe) {
    auto stripe_cfg = devinfo.settings.stripe_config();
    if (stripe_cfg.empty()) {
      throw std::runtime_error("Selected device must have a stripe_config when PLAIDML_USE_STRIPE is enabled");
    }
    auto out_path = env::Get("PLAIDML_STRIPE_OUTPUT");
    lang::RunInfo runinfo;
    runinfo.program = parsed;
    runinfo.input_shapes = inputs;
    runinfo.output_shapes = outputs;
    runinfo.program_name = "stripe_program";
    for (const auto& kvp : inputs) {
      if (kvp.second.is_const) {
        runinfo.const_inputs.emplace(kvp.first);
      }
    }
    return codegen::GenerateProgram(runinfo, stripe_cfg, out_path, const_bufs);
  }

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

Program::Program(                                             //
    const context::Context& ctx,                              //
    const tile::proto::Program& program,                      //
    const std::shared_ptr<DevInfo>& devinfo,                  //
    const std::shared_ptr<Scheduler>& scheduler,              //
    const std::shared_ptr<MemStrategy>& output_mem_strategy,  //
    const std::shared_ptr<MemStrategy>& tmp_mem_strategy,     //
    hal::Memory* tmp_memory,                                  //
    const lang::TileOptimizer& optimizer,                     //
    ConstBufferManager* const_bufs)
    : devinfo_{devinfo},  //
      output_mem_strategy_{output_mem_strategy},
      tmp_mem_strategy_{tmp_mem_strategy},
      num_runs_{0} {
  // TODO: Make this path asynchronous.
  // Asynchronous programming is a little tricky in this case, since if we compile asynchronously, the
  // compilation may not be complete when we're first asked to run a program, which means we'd need to save the run
  // request and mark the output buffers somehow to indicate that they're not just not ready due to hal events, they're
  // not ready due to compilation futures (which can't be waited on simultaneously).  There's a relatively
  // straightforward dataflow-ish way to describe the resulting system, but it's more complicated than just
  // compiling everything synchronously, so we just do everything synchronously for now.

  kernel_list_ = CompileProgram(program, *devinfo_.get(), optimizer, const_bufs);
  const_bufs_ = const_bufs->buffers;

  Initialize(ctx, program, scheduler);
}

Program::Program(                                             //
    const context::Context& ctx,                              //
    const std::shared_ptr<stripe::Program>& stripe,           //
    const std::string& target,                                //
    const std::shared_ptr<DevInfo>& devinfo,                  //
    const std::shared_ptr<Scheduler>& scheduler,              //
    const std::shared_ptr<MemStrategy>& output_mem_strategy,  //
    const std::shared_ptr<MemStrategy>& tmp_mem_strategy,     //
    hal::Memory* tmp_memory,                                  //
    ConstBufferManager* const_bufs)
    : devinfo_{devinfo},  //
      output_mem_strategy_{output_mem_strategy},
      tmp_mem_strategy_{tmp_mem_strategy},
      num_runs_{0} {
  auto out_path = env::Get("PLAIDML_STRIPE_OUTPUT");
  kernel_list_ = codegen::GenerateProgram(stripe, target, out_path, const_bufs);
  const_bufs_ = const_bufs->buffers;

  tile::proto::Program program;
  *program.mutable_inputs() = IntoProtoInput(stripe->input_shapes);
  *program.mutable_outputs() = IntoProtoOutput(stripe->output_shapes);
  Initialize(ctx, program, scheduler);
}

void Program::Initialize(          //
    const context::Context& ctx,   //
    tile::proto::Program program,  //
    const std::shared_ptr<Scheduler>& scheduler) {
  if (!devinfo_->dev->compiler() || !devinfo_->dev->executor()) {
    // TODO: Implement a mechanism for providing a pre-compiled program.
    throw error::Unavailable{"The requested device is unavailable for running Tile programs"};
  }
  memory_ = (devinfo_->dev->executor() && devinfo_->dev->executor()->device_memory()
                 ? devinfo_->dev->executor()->device_memory()
                 : devinfo_->devset->host_memory());

  for (const auto& kvp : const_bufs_) {
    if (!program.inputs().count(kvp.first)) {
      auto shape = kernel_list_.types.at(kvp.first);
      vertexai::tile::proto::ProgramInput input;
      (*input.mutable_shape()) = IntoProto(shape);
      (*program.mutable_inputs())[kvp.first] = input;
    }
  }

  context::Activity activity{ctx, "tile::local_machine::Compile"};
  auto lib = devinfo_->dev->compiler()->Build(activity.ctx(), kernel_list_.kernels, devinfo_->settings).get();
  executable_ = devinfo_->dev->executor()->Prepare(lib.get()).get();
  schedule_ = scheduler->BuildSchedule(program, kernel_list_);

  if (activity.ctx().is_logging_events()) {
    hal::proto::CompilationInfo cinfo;
    for (auto kernel : kernel_list_.kernels) {
      (*cinfo.mutable_kernels())[kernel.kname] = kernel.info;
    }
    SummarizeSchedule(&cinfo, program, kernel_list_, schedule_);
    *(cinfo.mutable_program()) = program;
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

void Program::Release() {
  // Restore the available memory
  // TODO: could switch to RAll pattern later
  std::lock_guard<std::mutex> guard(mutex);
  // Make sure that releases are not more than runs
  if (num_runs_ > 0) {
    avail_mem += alloc_mem_;
    --num_runs_;
    cond_var.notify_one();
  }
}

boost::future<void> Program::Run(const context::Context& ctx,
                                 std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
                                 std::map<std::string, std::shared_ptr<tile::Buffer>> outputs) {
  IVLOG(2, "Program::Run>");

  // This is the first program instance. Initialize the available memory and sync variables.
  std::call_once(first_run, [&]() { avail_mem = MaxAvailableMemory(); });

  alloc_mem_ = TotalAllocSize(schedule_, memory_->ArenaBufferAlignment());
  if (alloc_mem_ <= MaxAvailableMemory()) {
    std::unique_lock<std::mutex> guard(mutex);
    // TODO: could be asynchronous later
    // Wait for enough memory
    cond_var.wait(guard, [&] { return alloc_mem_ <= avail_mem; });
    // Reduce the available memory
    avail_mem -= alloc_mem_;
    ++num_runs_;
  } else {
    throw std::runtime_error(
        str(boost::format("No enough memory for the current schedule: required %1%, available %2%") % alloc_mem_ %
            MaxAvailableMemory()));
  }

  IVLOG(2, "  Inputs:");
  for (const auto& kvp : inputs) {
    IVLOG(2, "    " << kvp.first << ": " << kvp.second);
  }
  IVLOG(2, "  Outputs:");
  for (const auto& kvp : outputs) {
    IVLOG(2, "    " << kvp.first << ": " << kvp.second);
  }
  std::map<std::string, std::shared_ptr<tile::Buffer>> rewrite_outputs;
  for (auto kvp : outputs) {
    rewrite_outputs.emplace(kernel_list_.var_rewrites.Lookup(kvp.first), std::move(kvp.second));
  }
  for (const auto& kvp : const_bufs_) {
    inputs[kvp.first] = kvp.second;
  }
  return RunRequest::Run(ctx, shared_from_this(), std::move(inputs), std::move(rewrite_outputs));
}

std::size_t Program::MaxAvailableMemory() { return memory_->size_goal() * kGoalMemPercentage; }

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
