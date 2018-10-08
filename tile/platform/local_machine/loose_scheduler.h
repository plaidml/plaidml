// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <cstddef>
#include <memory>

#include "tile/platform/local_machine/placer.h"
#include "tile/platform/local_machine/scheduler.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// A scheduler for devices with asynchronous command queues.
//
// This implementation starts with a linear schedule (i.e. the minimum
// memory usage achievable without swapping buffers to system memory),
// and then "loosens" it, adjusting step dependencies to allow the
// hardware to increase parallelism while attempting to stay below the
// provided memory usage goal.
class LooseScheduler final : public Scheduler {
 public:
  LooseScheduler(const std::shared_ptr<Placer>& placer, std::uint64_t size_goal);

  schedule::Schedule BuildSchedule(const tile::proto::Program& program, const lang::KernelList& kl) final;

  const char* name() const final;

 private:
  std::shared_ptr<Placer> placer_;
  std::uint64_t size_goal_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
