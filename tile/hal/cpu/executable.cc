// Copyright 2017, Vertex.AI. CONFIDENTIAL

#include "tile/hal/cpu/executable.h"

#include <llvm/ExecutionEngine/ExecutionEngine.h>

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>

#include <boost/asio.hpp>
#include <boost/asio/thread_pool.hpp>

#include "base/util/error.h"
#include "tile/hal/cpu/buffer.h"
#include "tile/hal/cpu/event.h"
#include "tile/hal/cpu/runtime.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {
namespace {

const char invoker_prefix_[] = "__invoke_";

boost::asio::thread_pool thread_pool_(std::thread::hardware_concurrency());

}  // namespace

Executable::Executable(std::vector<std::shared_ptr<llvm::ExecutionEngine>> engines, std::vector<lang::KernelInfo> kis)
    : engines_{engines}, kis_(kis) {}

std::shared_ptr<hal::Event> Executable::Run(const context::Context& ctx, std::size_t kidx,
                                            const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                            const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                            bool /* enable_profiling */) {
  context::Activity activity(ctx, "tile::hal::cpu::Kernel::Run");
  std::vector<std::shared_ptr<hal::Buffer>> param_refs{params};
  auto deps = Event::WaitFor(dependencies);
  auto evt = deps.then([params = std::move(param_refs), act = std::move(activity), engine = engines_[kidx],
                        invoker_name = InvokerName(kis_[kidx].kname),
                        gwork = kis_[kidx].gwork](decltype(deps) future) -> std::shared_ptr<hal::Result> {
    future.get();
    auto start = std::chrono::high_resolution_clock::now();
    // Get the base address for all of these buffers, populating an argument
    // array, which we will pass in to the kernel's main function.
    std::vector<void*> args(params.size());
    for (size_t i = 0; i < args.size(); ++i) {
      args[i] = Buffer::Downcast(params[i])->base();
    }
    void* argvec = args.data();
    uint64_t entrypoint = engine->getFunctionAddress(invoker_name);
    // Iterate through the grid coordinates specified for this kernel, invoking
    // the kernel function once for each. We'll create one thread per core and
    // run one loop in each thread, staggering kernel invocations accordingly.
    size_t iterations = gwork[0] * gwork[1] * gwork[2];
    lang::GridSize denom = {{gwork[2] * gwork[1], gwork[2], 1}};
    size_t cores = std::thread::hardware_concurrency();
    size_t threads = std::min(iterations, cores);

    // The condition variable will guard the completion count. Each worker
    // will increment the completion count, and we'll wait until it reaches
    // the number of threads before returning.
    std::mutex mutex;
    std::condition_variable cv;
    size_t completed = 0;

    for (size_t offset = 0; offset < threads; ++offset) {
      boost::asio::post(thread_pool_, [=, &mutex, &cv, &completed]() {
        for (size_t i = offset; i < iterations; i += threads) {
          lang::GridSize index;
          index[0] = i / denom[0] % gwork[0];
          index[1] = i / denom[1] % gwork[1];
          index[2] = i / denom[2] % gwork[2];
          ((void (*)(void*, lang::GridSize*))entrypoint)(argvec, &index);
        }
        {
          std::unique_lock<std::mutex> lock(mutex);
          ++completed;
          cv.notify_all();
        }
      });
    }
    {
      std::unique_lock<std::mutex> lock(mutex);
      while (completed < threads) {
        cv.wait(lock);
      }
    }

    return std::make_shared<Result>(act.ctx(), "tile::hal::cpu::Executing", start,
                                    std::chrono::high_resolution_clock::now());
  });
  return std::make_shared<cpu::Event>(std::move(evt));
}

std::string Executable::InvokerName(std::string kname) { return invoker_prefix_ + kname; }

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
