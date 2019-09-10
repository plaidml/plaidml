// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/buffer.h"

#include <memory>

#include "base/util/error.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

std::shared_ptr<Buffer> Buffer::Downcast(const std::shared_ptr<hal::Buffer>& buffer) {
  std::shared_ptr<Buffer> buf = std::dynamic_pointer_cast<Buffer>(buffer);
  return buf;
}

Buffer* Buffer::Downcast(hal::Buffer* buffer) {
  Buffer* buf = dynamic_cast<Buffer*>(buffer);
  return buf;
}

Buffer::Buffer(std::uint64_t size) : size_{size} {}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
