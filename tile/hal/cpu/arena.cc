// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cpu/arena.h"

#include "base/util/error.h"
#include "tile/hal/cpu/buffer.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

Arena::Arena(std::uint64_t size) { mem_.resize(size, '\0'); }

std::shared_ptr<hal::Buffer> Arena::MakeBuffer(std::uint64_t offset, std::uint64_t size) {
  if (mem_.size() < offset || mem_.size() < size || mem_.size() < (offset + size)) {
    throw error::OutOfRange{"Requesting memory outside arena bounds"};
  }
  return std::make_shared<Buffer>(shared_from_this(), mem_.data() + offset, size);
}

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
