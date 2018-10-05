// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>

#include "tile/platform/local_machine/placer.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// BlockPlacer assigns allocs to separate memory blocks, coalescing
// them based on the schedule's dependency graph.
class BlockPlacer final : public Placer {
 public:
  explicit BlockPlacer(std::size_t alignment);

  std::unique_ptr<Placement> PlaceSchedule(const tile::proto::Program& program,
                                           schedule::Schedule* schedule) const final;

 private:
  std::size_t alignment_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
