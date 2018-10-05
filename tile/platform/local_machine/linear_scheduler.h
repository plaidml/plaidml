// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>

#include "tile/platform/local_machine/placer.h"
#include "tile/platform/local_machine/scheduler.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// A simple linear scheduler: this implementation simply creates an
// allocation for every temporary in the program, and runs the
// resulting program synchronously.
//
// This scheduler is primarily intended for use with purely
// synchronous devices that do not benefit from asynchronous
// scheduling.
//
// This scheduler implementation is also useful for chasing down
// tricky scheduling correctness issues, and for validating that the
// scheduler framework basically works (although note that other
// schedulers might exercise different edge cases in the framework).
class LinearScheduler final : public Scheduler {
 public:
  explicit LinearScheduler(const std::shared_ptr<Placer>& placer);

  schedule::Schedule BuildSchedule(const tile::proto::Program& program, const lang::KernelList& kl) final;

  const char* name() const final;

 private:
  std::shared_ptr<Placer> placer_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
