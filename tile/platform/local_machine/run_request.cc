// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/run_request.h"

#include <unordered_set>
#include <utility>

#include "base/util/error.h"
#include "tile/platform/local_machine/shim.h"

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

// Runs the schedule for a particular program.
boost::future<std::vector<std::shared_ptr<hal::Result>>> RunSchedule(  //
    const context::Context& ctx, RunRequest* req, Shim* shim) {
  std::vector<std::shared_ptr<hal::Event>> deps;
  deps.resize(req->program()->schedule().steps.size());
  std::unordered_set<std::shared_ptr<hal::Event>> dep_set;

  for (const auto& step : req->program()->schedule().steps) {
    IVLOG(2, "Queueing s" << step.idx << ": " << step);
    std::vector<std::shared_ptr<hal::Event>> current_deps;
    std::vector<std::shared_ptr<hal::Buffer>> current_params;
    std::vector<std::shared_ptr<MemChunk>> current_dep_chunks;

    auto add_chunk_param = [shim, &current_deps](std::size_t sidx, schedule::Alloc* alloc) {
      std::shared_ptr<MemChunk> chunk = shim->LookupAlloc(sidx, alloc);
      chunk->deps()->GetReadDependencies(&current_deps);
      return chunk;
    };

    for (const auto& dep : step.deps) {
      current_deps.emplace_back(deps[dep->idx]);
    }
    current_params.reserve(step.outputs.size() + step.inputs.size());
    current_dep_chunks.reserve(step.outputs.size());
    for (const auto& out : step.outputs) {
      std::shared_ptr<MemChunk> chunk = add_chunk_param(step.idx, out.allocp);
      current_params.emplace_back(chunk->hal_buffer());
      if (out.add_dep) {
        current_dep_chunks.push_back(chunk);
      }
    }
    for (const auto& in : step.inputs) {
      current_params.emplace_back(add_chunk_param(step.idx, in)->hal_buffer());
    }
    std::shared_ptr<hal::Event> event;
    switch (step.tag) {
      case schedule::Step::Tag::kRun:
        // NOTE: VLOG_IS_ON(1) is needed here because LogResults depends on profiling
        // being enabled in order to print durations.
        event = req->program()->executable()->Run(ctx, step.kidx, current_params, current_deps,
                                                  ctx.is_logging_events() || VLOG_IS_ON(1));
        break;
      case schedule::Step::Tag::kCopy:
        if (current_params.size() != 2) {
          throw error::Internal{"Invalid parameter count for copy step s" + std::to_string(step.idx)};
        }
        event = req->program()->devinfo()->dev->executor()->Copy(ctx, current_params[1], 0, current_params[0], 0,
                                                                 step.byte_count, current_deps);
        break;
      default:
        throw error::Internal{"Invalid schedule step s" + std::to_string(step.idx)};
    }
    for (const auto& chunk : current_dep_chunks) {
      chunk->deps()->AddReadDependency(event);
    }
    dep_set.insert(event);
    for (const auto& dep : current_deps) {
      dep_set.erase(dep);
    }
    deps[step.idx] = std::move(event);
  }

  boost::future<std::vector<std::shared_ptr<hal::Result>>> results;

  if (!dep_set.size()) {
    results = boost::make_ready_future<std::vector<std::shared_ptr<hal::Result>>>();
  } else {
    std::vector<std::shared_ptr<hal::Event>> terminal_deps;
    terminal_deps.reserve(dep_set.size());
    for (const auto& dep : dep_set) {
      terminal_deps.emplace_back(dep);
    }
    results = req->program()->devinfo()->dev->executor()->WaitFor(std::move(terminal_deps));
  }
  if (ctx.is_logging_events() || VLOG_IS_ON(1)) {
    // We want to return results for *all* of the steps.
    std::vector<boost::shared_future<std::shared_ptr<hal::Result>>> dep_futures;
    for (const auto& dep : deps) {
      dep_futures.emplace_back(dep->GetFuture());
    }
    results = results.then(
        [dep_futures = std::move(dep_futures)](boost::future<std::vector<std::shared_ptr<hal::Result>>> r) {
          r.get();
          // N.B. All of the step futures should be ready.
          std::vector<std::shared_ptr<hal::Result>> results;
          for (auto& dep : dep_futures) {
            results.push_back(dep.get());
          }
          return results;
        });
  }
  return results;
}

}  // namespace

boost::future<void> RunRequest::Run(          //
    const context::Context& ctx,              //
    const std::shared_ptr<Program>& program,  //
    std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
    std::map<std::string, std::shared_ptr<tile::Buffer>> outputs) {
  LogRequest(program, inputs, outputs);

  RunRequest req{program};

  context::Activity running{ctx, "tile::local_machine::Program::Run"};
  boost::future<void> complete;
  auto shim = std::make_unique<Shim>(running.ctx(), program, std::move(inputs), std::move(outputs));

  {
    context::Activity queueing{running.ctx(), "tile::local_machine::Program::Enqueue"};
    boost::future<std::vector<std::shared_ptr<hal::Result>>> results;

    try {
      results = RunSchedule(queueing.ctx(), &req, shim.get());
    } catch (...) {
      shim->SetLaunchException(std::current_exception());
      // If this happens, it's probably an OOM.
      // TODO: Synchronize with the HAL to ensure all ongoing activity is complete,
      // so that we can safely release any memory we're holding onto.
      return boost::make_ready_future();
    }
    shim->OnLaunchSuccess();
    complete = req.LogResults(queueing.ctx(), std::move(results));
  }

  // Keep the shim and activity referenced until the program is complete.
  // N.B. It's important to keep the shim referenced because it's the thing that's actually holding
  // onto all of our chunk references; if those go away, unfortunate things happen.
  return complete.then([shim = std::move(shim), running = std::move(running)](decltype(complete) fut) {  //
    fut.get();
  });
}

void RunRequest::LogRequest(                  //
    const std::shared_ptr<Program>& program,  //
    const std::map<std::string, std::shared_ptr<tile::Buffer>>& inputs,
    const std::map<std::string, std::shared_ptr<tile::Buffer>>& outputs) {
  VLOG(1) << "Running program " << program;
  if (VLOG_IS_ON(2)) {
    for (const auto& it : inputs) {
      std::shared_ptr<Buffer> buffer = Buffer::Downcast(it.second, program->devinfo());
      std::shared_ptr<MemChunk> chunk = buffer->chunk();
      if (chunk) {
        VLOG(2) << "Input  " << it.first << " -> Buffer " << buffer.get() << " -> HAL Buffer "
                << chunk->hal_buffer().get() << ", size=" << chunk->size() << " bytes";
      } else {
        VLOG(2) << "Input  " << it.first << " -> Buffer " << buffer.get() << " -> No chunk, size=" << buffer->size()
                << " bytes";
      }
    }
    for (const auto& it : outputs) {
      std::shared_ptr<Buffer> buffer = Buffer::Downcast(it.second, program->devinfo());
      std::shared_ptr<MemChunk> chunk = buffer->chunk();
      if (chunk) {
        VLOG(2) << "Output " << it.first << " -> Buffer " << buffer.get() << " -> HAL Buffer "
                << chunk->hal_buffer().get() << ", size=" << chunk->size() << " bytes";
      } else {
        VLOG(2) << "Output " << it.first << " -> Buffer " << buffer.get() << " -> No chunk, size=" << buffer->size()
                << " bytes";
      }
    }
  }
}

boost::future<void> RunRequest::LogResults(  //
    const context::Context& ctx,             //
    boost::future<std::vector<std::shared_ptr<hal::Result>>> results) {
  context::Context ctx_copy{ctx};
  return results.then([ctx = std::move(ctx_copy)](decltype(results) future) {
    auto results = future.get();
    if (VLOG_IS_ON(1) || ctx.is_logging_events()) {
      std::chrono::high_resolution_clock::duration total{std::chrono::high_resolution_clock::duration::zero()};
      for (const auto& result : results) {
        total += result->GetDuration();
        result->LogStatistics();
      }
      VLOG(1) << "Total program execution duration: " << total.count();
    }
  });
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
