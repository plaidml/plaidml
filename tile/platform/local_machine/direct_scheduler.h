// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>

#include "tile/platform/local_machine/scheduler.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// DirectScheduler
class DirectScheduler final : public Scheduler {
 public:
  Schedule BuildSchedule(const tile::proto::Program& program, const lang::KernelList& kl) final;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
