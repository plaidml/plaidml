// Copyright 2017, Intel Corporation.

#pragma once

#include <memory>

#include "tile/platform/local_machine/placer.h"
#include "tile/platform/local_machine/scheduler.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// The transitive dependency scheduler starts from a simple linear
// schedule, then sets each step's dependencies based on its inputs.
// It works well for asynchronous devices that have more memory than
// the program requires.
//
class TransitiveDepScheduler final : public Scheduler {
 public:
  explicit TransitiveDepScheduler(const std::shared_ptr<Placer>& placer, std::size_t max_in_flight);

  schedule::Schedule BuildSchedule(const tile::proto::Program& program, const lang::KernelList& kl) final;

  const char* name() const final;

 private:
  std::shared_ptr<Placer> placer_;
  std::size_t max_in_flight_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
