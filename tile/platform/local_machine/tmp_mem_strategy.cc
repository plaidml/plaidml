// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/tmp_mem_strategy.h"

#include <exception>
#include <utility>

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

// A MemChunk implementation that frees its underlying memory to a MemCache when the chunk is deleted.
class TmpMemChunk final : public MemChunk {
 public:
  TmpMemChunk(std::uint64_t size, const std::shared_ptr<MemCache>& mem_cache, std::shared_ptr<hal::Buffer> hal_buffer);
  virtual ~TmpMemChunk();

  std::uint64_t size() const final;
  boost::future<std::unique_ptr<View>> MapCurrent(const context::Context& ctx) final;
  std::unique_ptr<View> MapDiscard(const context::Context& ctx) final;
  std::shared_ptr<MemDeps> deps() final;
  std::shared_ptr<hal::Buffer> hal_buffer() final;

 private:
  std::uint64_t size_;
  std::shared_ptr<MemCache> mem_cache_;
  std::shared_ptr<hal::Buffer> hal_buffer_;
  std::shared_ptr<MemDeps> deps_;
};

TmpMemChunk::TmpMemChunk(std::uint64_t size, const std::shared_ptr<MemCache>& mem_cache,
                         std::shared_ptr<hal::Buffer> hal_buffer)
    : size_{size}, mem_cache_{mem_cache}, hal_buffer_{hal_buffer}, deps_{std::make_shared<MemDeps>()} {}

TmpMemChunk::~TmpMemChunk() { mem_cache_->Free(size_, std::move(hal_buffer_)); }

std::uint64_t TmpMemChunk::size() const { return size_; }

boost::future<std::unique_ptr<View>> TmpMemChunk::MapCurrent(const context::Context& ctx) {
  throw std::runtime_error("unable to map a temporary memory buffer");
}

std::unique_ptr<View> TmpMemChunk::MapDiscard(const context::Context& ctx) {
  throw std::runtime_error("unable to map a temporary memory buffer");
}

std::shared_ptr<MemDeps> TmpMemChunk::deps() { return deps_; }

std::shared_ptr<hal::Buffer> TmpMemChunk::hal_buffer() { return hal_buffer_; }

}  // namespace

TmpMemStrategy::TmpMemStrategy(const std::shared_ptr<DevInfo>& devinfo, hal::Memory* source)
    : devinfo_{devinfo}, source_{source}, cache_{std::make_shared<MemCache>()} {
  if (!source_) {
    throw std::logic_error{"The temporary memory management strategy requires memory"};
  }
}

std::shared_ptr<MemChunk> TmpMemStrategy::MakeChunk(const context::Context& ctx, std::uint64_t size) const {
  auto hal_buffer = cache_->TryAlloc(size);
  if (!hal_buffer) {
    auto buffer = source_->MakeBuffer(size, hal::BufferAccessMask::DEVICE_RW);
    hal_buffer = buffer;
  }
  return std::make_shared<TmpMemChunk>(size, cache_, std::move(hal_buffer));
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
