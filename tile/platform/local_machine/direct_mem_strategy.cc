// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/direct_mem_strategy.h"

#include <stdexcept>
#include <utility>

#include "base/util/compat.h"

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

// Local classes

class DirectMemView final : public View {
 public:
  DirectMemView(const context::Context& ctx, std::shared_ptr<MemDeps> deps, void* data, std::size_t size,
                std::shared_ptr<hal::Buffer> mem);
  ~DirectMemView();

  void WriteBack(const context::Context& ctx) final;

 private:
  context::Context unmap_ctx_;
  std::shared_ptr<MemDeps> deps_;
  std::shared_ptr<hal::Buffer> mem_;
};

class DirectMemChunk final : public MemChunk {
 public:
  DirectMemChunk(const context::Context& ctx, const std::shared_ptr<DevInfo>& devinfo, std::uint64_t size,
                 hal::Memory* source);

  // Buffer implementation
  boost::future<std::unique_ptr<View>> MapCurrent(const context::Context& ctx) final;
  std::unique_ptr<View> MapDiscard(const context::Context& ctx) final;
  std::uint64_t size() const final { return size_; }

  // MemChunk implementation
  std::shared_ptr<MemDeps> deps() final { return deps_; }
  std::shared_ptr<hal::Buffer> hal_buffer() final { return mem_; }

 private:
  std::uint64_t size_;
  std::shared_ptr<DevInfo> devinfo_;
  std::shared_ptr<MemDeps> deps_;
  std::shared_ptr<hal::Buffer> mem_;
};

// Implementation

DirectMemView::DirectMemView(const context::Context& ctx, std::shared_ptr<MemDeps> deps, void* data, std::size_t size,
                             std::shared_ptr<hal::Buffer> mem)
    : View(static_cast<char*>(data), size), unmap_ctx_{ctx}, deps_{std::move(deps)}, mem_{std::move(mem)} {}

DirectMemView::~DirectMemView() {
  if (data()) {
    deps_->AddReadDependency(mem_->Unmap(unmap_ctx_));
  }
}

void DirectMemView::WriteBack(const context::Context& ctx) {
  deps_->AddReadDependency(mem_->Unmap(ctx));
  set_contents(nullptr, 0);
}

DirectMemChunk::DirectMemChunk(const context::Context& ctx, const std::shared_ptr<DevInfo>& devinfo, std::uint64_t size,
                               hal::Memory* source)
    : size_{size}, devinfo_{devinfo}, deps_{std::make_shared<MemDeps>()} {
  mem_ = source->MakeBuffer(size_, hal::BufferAccessMask::ALL);
}

boost::future<std::unique_ptr<View>> DirectMemChunk::MapCurrent(const context::Context& ctx) {
  context::Context ctx_copy{ctx};
  std::vector<std::shared_ptr<hal::Event>> deps;
  deps_->GetReadDependencies(&deps);
  return mem_->MapCurrent(deps).then([ctx = std::move(ctx_copy), deps = deps_, size = size_,
                                      mem = mem_](boost::future<void*> data_future) mutable -> std::unique_ptr<View> {
    void* data = data_future.get();
    return compat::make_unique<DirectMemView>(ctx, std::move(deps), data, size, std::move(mem));
  });
}

std::unique_ptr<View> DirectMemChunk::MapDiscard(const context::Context& ctx) {
  std::vector<std::shared_ptr<hal::Event>> deps;
  deps_->GetReadDependencies(&deps);
  void* data = mem_->MapDiscard(deps).get();
  return compat::make_unique<DirectMemView>(ctx, deps_, data, size_, mem_);
}

}  // namespace

DirectMemStrategy::DirectMemStrategy(const std::shared_ptr<DevInfo>& devinfo, hal::Memory* source)
    : devinfo_{devinfo}, source_{source} {
  if (!source_) {
    throw std::logic_error{"The direct memory management strategy requires source memory"};
  }
}

std::shared_ptr<MemChunk> DirectMemStrategy::MakeChunk(const context::Context& ctx, std::uint64_t size) const {
  return std::make_shared<DirectMemChunk>(ctx, devinfo_, size, source_);
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
