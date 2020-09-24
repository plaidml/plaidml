// Copyright 2020 Intel Corporation

#include "pmlc/rt/timed_executable.h"

#include <chrono>
#include <utility>

#include "pmlc/util/logging.h"

namespace pmlc::rt {
namespace {

// A trivial stopwatch class.
//
// TODO: Consider using std::chrono::high_resolution_clock.  Also consider
// detecting core switch events and hyperthreading, and notifying the user if
// CPU frequency scaling is enabled.  (Realistically, this should probably be
// done as a test, completely external to the runtime.)
struct StopWatch {
  using fp_milliseconds =
      std::chrono::duration<double, std::chrono::milliseconds::period>;

  void start() { startTime = std::chrono::steady_clock::now(); }

  void stop() { stopTime = std::chrono::steady_clock::now(); }

  double delta_ms() {
    return std::chrono::duration_cast<fp_milliseconds>(stopTime - startTime)
        .count();
  }

  std::chrono::steady_clock::time_point startTime;
  std::chrono::steady_clock::time_point stopTime;
};

class TimedExecutable final : public Executable {
public:
  explicit TimedExecutable(std::unique_ptr<Executable> base)
      : base{std::move(base)} {}

  void invoke() final {
    StopWatch stopWatch;
    stopWatch.start();
    base->invoke();
    stopWatch.stop();
    IVLOG(1, "Executable time: " << stopWatch.delta_ms() << "ms");
  }

private:
  std::unique_ptr<Executable> base;
};

} // namespace

// Returns an Executable implementation that times how long it takes to perform
// an inference.
std::unique_ptr<Executable>
makeTimedExecutable(std::unique_ptr<Executable> base) {
  return std::make_unique<TimedExecutable>(std::move(base));
}

} // namespace pmlc::rt
