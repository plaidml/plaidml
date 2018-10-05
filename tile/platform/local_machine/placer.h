// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <cstdint>
#include <memory>

#include "tile/base/schedule.h"
#include "tile/proto/tile.pb.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// A Placement is an arrangement of memory allocations for a schedule.
class Placement {
 public:
  virtual ~Placement() {}

  // The memory required by this placement.
  virtual std::uint64_t device_memory_bytes() const = 0;

  // Updates the placed schedule with the allocations created by the
  // placement.  N.B. This must only be called once per schedule.
  virtual void Apply() = 0;
};

// A Placer determines where to place memory allocations for a
// schedule, generating a Placement.
class Placer {
 public:
  virtual ~Placer() {}

  // Returns a Placement for the memory required by a schedule.
  virtual std::unique_ptr<Placement> PlaceSchedule(const tile::proto::Program& program,
                                                   schedule::Schedule* schedule) const = 0;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
