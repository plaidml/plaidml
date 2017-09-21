// Copyright 2017, Vertex.AI. CONFIDENTIAL

#pragma once

#include <memory>
#include <mutex>

#include "tile/platform/local_machine/devinfo.h"
#include "tile/platform/local_machine/mem_chunk.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// A handle to a chunk of memory.
// Note that the particular chunk referenced by a buffer may change over time.
class Buffer : public tile::Buffer, public std::enable_shared_from_this<Buffer> {
 public:
  // Casts a tile::Buffer to a Buffer for the indicated device, throwing an
  // exception if the supplied buffer is not a Buffer or if it was allocated on a
  // different device.
  static std::shared_ptr<Buffer> Upcast(const std::shared_ptr<tile::Buffer>& buffer,
                                        const std::shared_ptr<DevInfo>& devinfo);

  Buffer(const std::shared_ptr<DevInfo>& devinfo, std::shared_ptr<MemChunk> chunk);

  const std::shared_ptr<DevInfo>& devinfo() const { return devinfo_; }

  std::shared_ptr<MemChunk> chunk() const {
    std::lock_guard<std::mutex> lock{mu_};
    return chunk_;
  }

  // Buffer implementation.
  boost::future<std::unique_ptr<View>> MapCurrent(const context::Context& ctx) final;
  std::unique_ptr<View> MapDiscard(const context::Context& ctx) final;
  std::uint64_t size() const final;

  void RemapTo(const std::shared_ptr<MemChunk>& chunk);

 private:
  const std::shared_ptr<DevInfo> devinfo_;
  mutable std::mutex mu_;
  std::shared_ptr<MemChunk> chunk_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
