// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <cstddef>
#include <memory>

#include "tile/platform/local_machine/placer.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// NaivePlacer does no schedule dependency analysis or rewriting; it
// simply computes the memory required for the individual allocations
// given the supplied alignment.
class NaivePlacer final : public Placer {
 public:
  explicit NaivePlacer(std::size_t alignment);

  std::unique_ptr<Placement> PlaceSchedule(const tile::proto::Program& program,
                                           schedule::Schedule* schedule) const final;

 private:
  std::size_t alignment_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
