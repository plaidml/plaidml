// Copyright 2017, Vertex.AI.

#include "tile/platform/local_machine/run_request.h"

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

// Runs the schedule for a particular program.
class ScheduleRunner final : private StepVisitor {
 public:
  static boost::future<std::vector<std::shared_ptr<hal::Result>>> Run(const context::Context& ctx, RunRequest* req) {
    ScheduleRunner runner{ctx, req};
    runner.deps_.resize(req->program()->schedule().steps.size());
    for (const auto& step : req->program()->schedule().steps) {
      IVLOG(1, "Queueing s" << step->idx << ": " << *step);
      step->Accept(&runner);
    }

    return req->program()->devinfo()->dev->executor()->WaitFor(runner.deps_);
  }

 private:
  ScheduleRunner(const context::Context& ctx, RunRequest* req) : ctx_{ctx}, req_{req} {}

  void Visit(const RunStep& run) final {
    auto deps = InitDeps(run);
    std::vector<std::shared_ptr<hal::Buffer>> params;
    std::vector<std::shared_ptr<MemChunk>> dep_chunks;
    params.reserve(run.outputs.size() + run.inputs.size());
    dep_chunks.reserve(run.outputs.size());
    for (const auto& out : run.outputs) {
      std::shared_ptr<MemChunk> chunk = AddChunkParam(run.idx, out.allocp, &deps);
      params.emplace_back(chunk->hal_buffer());
      if (out.add_dep) {
        dep_chunks.push_back(chunk);
      }
    }
    for (const auto& in : run.inputs) {
      params.emplace_back(AddChunkParam(run.idx, in, &deps)->hal_buffer());
    }
    // NOTE: VLOG_IS_ON(1) is needed here because LogResults depends on profiling
    // being enabled in order to print durations.
    auto event =
        req_->program()->kernels()[run.kidx]->Run(ctx_, params, deps, ctx_.is_logging_events() || VLOG_IS_ON(1));
    for (const auto& chunk : dep_chunks) {
      chunk->deps()->AddReadDependency(event);
    }
    req_->AddKernelInfo(run.kidx, event);
    deps_[run.idx] = std::move(event);
  }

  void Visit(const CopyStep& copy) final {
    auto deps = InitDeps(copy);
    std::shared_ptr<MemChunk> from_chunk = AddChunkParam(copy.idx, copy.from, &deps);
    std::shared_ptr<MemChunk> to_chunk = AddChunkParam(copy.idx, copy.to.allocp, &deps);
    auto event = req_->program()->devinfo()->dev->executor()->Copy(ctx_, from_chunk->hal_buffer(), 0,
                                                                   to_chunk->hal_buffer(), 0, copy.byte_count, deps);
    if (copy.to.add_dep) {
      to_chunk->deps()->AddReadDependency(event);
    }
    deps_[copy.idx] = std::move(event);
  }

  std::shared_ptr<MemChunk> AddChunkParam(std::size_t sidx, AllocPtr alloc,
                                          std::vector<std::shared_ptr<hal::Event>>* deps) {
    std::shared_ptr<MemChunk> chunk = req_->shim()->LookupAlloc(sidx, alloc);
    auto extra_deps = chunk->deps()->GetReadDependencies();
    deps->insert(deps->end(), std::make_move_iterator(extra_deps.begin()), std::make_move_iterator(extra_deps.end()));
    return chunk;
  }

  std::vector<std::shared_ptr<hal::Event>> InitDeps(const Step& step) {
    std::vector<std::shared_ptr<hal::Event>> deps;
    for (const auto& dep : step.deps) {
      deps.emplace_back(deps_[(*dep)->idx]);
    }
    return deps;
  }

  context::Context ctx_;
  RunRequest* req_;
  std::vector<std::shared_ptr<hal::Event>> deps_;
};

}  // namespace

boost::future<void> RunRequest::Run(const context::Context& ctx, const Program* program,
                                    std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
                                    std::map<std::string, std::shared_ptr<tile::Buffer>> outputs) {
  LogRequest(program, inputs, outputs);

  auto shim = compat::make_unique<Shim>(ctx, program, std::move(inputs), std::move(outputs));

  RunRequest req{program, std::move(shim)};

  context::Activity running{ctx, "tile::local_machine::Program::Run"};
  boost::future<void> complete;

  {
    context::Activity queueing{running.ctx(), "tile::local_machine::Program::Enqueue"};
    boost::future<std::vector<std::shared_ptr<hal::Result>>> results;
    try {
      results = ScheduleRunner::Run(queueing.ctx(), &req);
    } catch (...) {
      req.shim()->SetLaunchException(std::current_exception());
    }
    complete = req.LogResults(queueing.ctx(), std::move(results));
  }

  // Keep the request and activity referenced until the program is complete.
  return complete.then([ req = std::move(req), running = std::move(running) ](decltype(complete) fut) { fut.get(); });
}

void RunRequest::AddKernelInfo(std::size_t kidx, std::shared_ptr<hal::Event> event) {
  const lang::KernelInfo& ki = program_->kernel_list().kernels[kidx];
  kernel_log_info_.emplace_back(KernelLogInfo{std::move(event), ki.kname, ki.tot_bytes, ki.tot_flops});
}

RunRequest::RunRequest(const Program* program, std::unique_ptr<Shim> shim) : program_{program}, shim_{std::move(shim)} {
  kernel_log_info_.reserve(program_->kernels().size());
}

void RunRequest::LogRequest(const Program* program, const std::map<std::string, std::shared_ptr<tile::Buffer>>& inputs,
                            const std::map<std::string, std::shared_ptr<tile::Buffer>>& outputs) {
  if (VLOG_IS_ON(1)) {
    VLOG(1) << "Running program " << program;
    for (const auto& it : inputs) {
      std::shared_ptr<MemChunk> chunk = Buffer::Downcast(it.second, program->devinfo())->chunk();
      VLOG(1) << "Input  " << it.first << " -> Buffer " << it.second.get() << " -> HAL Buffer "
              << chunk->hal_buffer().get() << ", size=" << chunk->size() << " bytes";
    }
    for (const auto& it : outputs) {
      std::shared_ptr<MemChunk> chunk = Buffer::Downcast(it.second, program->devinfo())->chunk();
      VLOG(1) << "Output " << it.first << " -> Buffer " << it.second.get() << " -> HAL Buffer "
              << chunk->hal_buffer().get() << ", size=" << chunk->size() << " bytes";
    }
  }
}

boost::future<void> RunRequest::LogResults(const context::Context& ctx,
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
