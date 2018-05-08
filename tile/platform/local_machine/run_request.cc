// Copyright 2017, Vertex.AI.

#include "tile/platform/local_machine/run_request.h"

#include <unordered_set>

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

// Runs the schedule for a particular program.
class ScheduleRunner final : private StepVisitor {
 public:
  static boost::future<std::vector<std::shared_ptr<hal::Result>>> Run(const context::Context& ctx, RunRequest* req) {
    ScheduleRunner runner{ctx, req};
    auto size = req->program()->schedule().steps.size();
    runner.deps_.resize(size);
    std::vector<std::shared_ptr<hal::Event>> terminal_deps;
    terminal_deps.reserve(size);  // Pre-allocate for the worst case.
    try {
      for (const auto& step : req->program()->schedule().steps) {
        IVLOG(1, "Queueing s" << step->idx << ": " << *step);
        step->Accept(&runner);
      }
    } catch (...) {
      req->shim()->SetLaunchException(std::current_exception());
    }

    boost::future<std::vector<std::shared_ptr<hal::Result>>> results;

    if (!runner.dep_set_.size()) {
      results = boost::make_ready_future<std::vector<std::shared_ptr<hal::Result>>>();
    } else {
      for (const auto& dep : runner.dep_set_) {
        terminal_deps.emplace_back(dep);
      }
      results = req->program()->devinfo()->dev->executor()->WaitFor(terminal_deps);
    }
    if (ctx.is_logging_events() || VLOG_IS_ON(1)) {
      // We want to return results for *all* of the steps.
      std::vector<boost::shared_future<std::shared_ptr<hal::Result>>> dep_futures;
      for (const auto& dep : runner.deps_) {
        dep_futures.emplace_back(dep->GetFuture());
      }
      results = results.then([dep_futures=std::move(dep_futures)](boost::future<std::vector<std::shared_ptr<hal::Result>>> r) {
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

 private:
  ScheduleRunner(const context::Context& ctx, RunRequest* req) : ctx_{ctx}, req_{req} {}

  void Visit(const RunStep& run) final {
    InitDeps(run);
    current_params_.reserve(run.outputs.size() + run.inputs.size());
    current_dep_chunks_.reserve(run.outputs.size());
    for (const auto& out : run.outputs) {
      std::shared_ptr<MemChunk> chunk = AddChunkParam(run.idx, out.allocp);
      current_params_.emplace_back(chunk->hal_buffer());
      if (out.add_dep) {
        current_dep_chunks_.push_back(chunk);
      }
    }
    for (const auto& in : run.inputs) {
      current_params_.emplace_back(AddChunkParam(run.idx, in)->hal_buffer());
    }
    // NOTE: VLOG_IS_ON(1) is needed here because LogResults depends on profiling
    // being enabled in order to print durations.
    auto event =
        req_->program()->kernels()[run.kidx]->Run(ctx_, current_params_, current_deps_, ctx_.is_logging_events() || VLOG_IS_ON(1));
    for (const auto& chunk : current_dep_chunks_) {
      chunk->deps()->AddReadDependency(event);
    }
    dep_set_.insert(event);
    for (const auto& dep : current_deps_) {
      dep_set_.erase(dep);
    }
    deps_[run.idx] = std::move(event);
    current_deps_.resize(0);
    current_params_.resize(0);
    current_dep_chunks_.resize(0);
  }

  void Visit(const CopyStep& copy) final {
    InitDeps(copy);
    std::shared_ptr<MemChunk> from_chunk = AddChunkParam(copy.idx, copy.from);
    std::shared_ptr<MemChunk> to_chunk = AddChunkParam(copy.idx, copy.to.allocp);
    auto event = req_->program()->devinfo()->dev->executor()->Copy(ctx_, from_chunk->hal_buffer(), 0,
                                                                   to_chunk->hal_buffer(), 0, copy.byte_count, current_deps_);
    if (copy.to.add_dep) {
      to_chunk->deps()->AddReadDependency(event);
    }
    dep_set_.insert(event);
    for (const auto& dep : current_deps_) {
      dep_set_.erase(dep);
    }
    deps_[copy.idx] = std::move(event);
    current_deps_.resize(0);
  }

  std::shared_ptr<MemChunk> AddChunkParam(std::size_t sidx, AllocPtr alloc) {
    std::shared_ptr<MemChunk> chunk = req_->shim()->LookupAlloc(sidx, alloc);
    chunk->deps()->GetReadDependencies(&current_deps_);
    return chunk;
  }

  void InitDeps(const Step& step) {
    for (const auto& dep : step.deps) {
      current_deps_.emplace_back(deps_[(*dep)->idx]);
    }
  }

  context::Context ctx_;
  RunRequest* req_;
  std::vector<std::shared_ptr<hal::Event>> deps_;
  std::unordered_set<std::shared_ptr<hal::Event>> dep_set_;

  // Used while issuing work.
  std::vector<std::shared_ptr<hal::Event>> current_deps_;
  std::vector<std::shared_ptr<hal::Buffer>> current_params_;
  std::vector<std::shared_ptr<MemChunk>> current_dep_chunks_;
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
      // If this happens, it's probably an OOM.
      // TODO: Synchronize with the HAL to ensure all ongoing activity is complete,
      // so that we can safely release any memory we're holding onto.
      return boost::make_ready_future();
    }
    complete = req.LogResults(queueing.ctx(), std::move(results));
  }

  // Keep the request and activity referenced until the program is complete.
  return complete.then([ req = std::move(req), running = std::move(running) ](decltype(complete) fut) { fut.get(); });
}

RunRequest::RunRequest(const Program* program, std::unique_ptr<Shim> shim) : program_{program}, shim_{std::move(shim)} {}

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
