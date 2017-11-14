// Copyright 2017, Vertex.AI.

#include "tile/platform/local_machine/buffer.h"

#include <utility>

#include "base/util/error.h"
#include "base/util/logging.h"

namespace vertexai {
namespace tile {
namespace local_machine {

std::shared_ptr<Buffer> Buffer::Downcast(const std::shared_ptr<tile::Buffer>& buffer,
                                         const std::shared_ptr<DevInfo>& devinfo) {
  auto result = std::dynamic_pointer_cast<Buffer>(buffer);
  if (!result) {
    throw error::InvalidArgument("incompatible buffer type");
  }
  if (result->devinfo_ != devinfo) {
    throw error::InvalidArgument("incompatible buffer for device");
  }
  return result;
}

Buffer::Buffer(const std::shared_ptr<DevInfo>& devinfo, std::shared_ptr<MemChunk> chunk)
    : devinfo_{devinfo}, chunk_{std::move(chunk)} {}

boost::future<std::unique_ptr<View>> Buffer::MapCurrent(const context::Context& ctx) {
  return chunk()->MapCurrent(ctx);
}

std::unique_ptr<View> Buffer::MapDiscard(const context::Context& ctx) { return chunk()->MapDiscard(ctx); }

std::uint64_t Buffer::size() const { return chunk()->size(); }

void Buffer::RemapTo(const std::shared_ptr<MemChunk>& chunk) {
  if (size() != chunk->size()) {
    throw std::runtime_error("The requested buffer remapping required a change in buffer size");
  }
  std::lock_guard<std::mutex> lock{mu_};
  chunk_ = chunk;
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
