// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/naive_placer.h"

#include "base/util/compat.h"
#include "base/util/logging.h"

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

class NaivePlacement final : public Placement {
 public:
  explicit NaivePlacement(std::uint64_t sum) : sum_{sum} {}

  std::uint64_t device_memory_bytes() const final { return sum_; }

  void Apply() final { IVLOG(1, "Naive placer: Schedule uses " << sum_ << " bytes of device memory"); }

 private:
  const std::uint64_t sum_;
};

}  // namespace

NaivePlacer::NaivePlacer(std::size_t alignment) : alignment_{alignment} {}

std::unique_ptr<Placement> NaivePlacer::PlaceSchedule(const tile::proto::Program& program,
                                                      schedule::Schedule* schedule) const {
  std::uint64_t sum = 0;
  for (const auto& alloc : schedule->allocs) {
    sum += ((alloc.byte_size + alignment_ - 1) / alignment_) * alignment_;
  }
  return compat::make_unique<NaivePlacement>(sum);
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
