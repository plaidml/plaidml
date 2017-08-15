// Copyright 2017, Vertex.AI.

#include "tile/platform/local_machine/copy_mem_strategy.h"

#include <stdexcept>
#include <utility>

#include "base/util/compat.h"

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

// Local classes

class CopyMemView final : public View {
 public:
  CopyMemView(const context::Context& ctx, const std::shared_ptr<DevInfo>& devinfo, std::shared_ptr<MemDeps> deps,
              void* data, std::size_t size, std::shared_ptr<hal::Buffer> dev_mem,
              std::shared_ptr<hal::Buffer> host_mem);
  ~CopyMemView();

  void WriteBack(const context::Context& ctx) final;

 private:
  context::Context unmap_ctx_;
  std::shared_ptr<DevInfo> devinfo_;
  std::shared_ptr<MemDeps> deps_;
  std::shared_ptr<hal::Buffer> dev_mem_;
  std::shared_ptr<hal::Buffer> host_mem_;
};

class CopyMemChunk final : public MemChunk {
 public:
  CopyMemChunk(const context::Context& ctx, const std::shared_ptr<DevInfo>& devinfo, std::uint64_t size);

  // Buffer implementation
  boost::future<std::unique_ptr<View>> MapCurrent(const context::Context& ctx) final;
  std::unique_ptr<View> MapDiscard(const context::Context& ctx) final;
  std::uint64_t size() const final { return size_; }

  // MemChunk implementation
  std::shared_ptr<MemDeps> deps() final { return deps_; }
  std::shared_ptr<hal::Buffer> hal_buffer() final { return dev_mem_; }

 private:
  std::uint64_t size_;
  std::shared_ptr<DevInfo> devinfo_;
  std::shared_ptr<MemDeps> deps_;
  std::shared_ptr<hal::Buffer> dev_mem_;
};

// Implementation

CopyMemView::CopyMemView(const context::Context& ctx, const std::shared_ptr<DevInfo>& devinfo,
                         // cppcheck-suppress passedByValue  // NOLINT
                         std::shared_ptr<MemDeps> deps, void* data, std::size_t size,
                         std::shared_ptr<hal::Buffer> dev_mem, std::shared_ptr<hal::Buffer> host_mem)
    : View(static_cast<char*>(data), size),
      unmap_ctx_{ctx},
      devinfo_{devinfo},
      deps_{std::move(deps)},
      dev_mem_{std::move(dev_mem)},
      host_mem_{std::move(host_mem)} {}

CopyMemView::~CopyMemView() {
  if (data()) {
    deps_->AddReadDependency(host_mem_->Unmap(unmap_ctx_));
  }
}

void CopyMemView::WriteBack(const context::Context& ctx) {
  auto unmapped = host_mem_->Unmap(ctx);
  auto sz = size();
  set_contents(nullptr, 0);
  auto copied = devinfo_->dev->executor()->Copy(ctx, host_mem_, 0, dev_mem_, 0, sz, {unmapped});
  deps_->AddReadDependency(std::move(copied));
}

CopyMemChunk::CopyMemChunk(const context::Context& ctx, const std::shared_ptr<DevInfo>& devinfo, std::uint64_t size)
    : size_{size}, devinfo_{devinfo}, deps_{std::make_shared<MemDeps>()} {
  dev_mem_ = devinfo_->dev->executor()->device_memory()->MakeBuffer(
      size, hal::BufferAccessMask::DEVICE_READABLE | hal::BufferAccessMask::DEVICE_WRITEABLE);
}

boost::future<std::unique_ptr<View>> CopyMemChunk::MapCurrent(const context::Context& ctx) {
  auto host_mem = devinfo_->devset->host_memory()->MakeBuffer(
      size_, hal::BufferAccessMask::HOST_READABLE | hal::BufferAccessMask::HOST_WRITEABLE);
  auto copied = devinfo_->dev->executor()->Copy(ctx, dev_mem_, 0, host_mem, 0, size(), deps_->GetReadDependencies());
  return host_mem->MapCurrent({copied}).then([
    ctx = context::Context{ctx}, devinfo = devinfo_, deps = deps_, size = size_, dev_mem = dev_mem_, host_mem = host_mem
  ](boost::future<void*> data_future) mutable->std::unique_ptr<View> {
    void* data = data_future.get();
    return compat::make_unique<CopyMemView>(ctx, devinfo, deps, data, size, dev_mem, std::move(host_mem));
  });
}

std::unique_ptr<View> CopyMemChunk::MapDiscard(const context::Context& ctx) {
  auto host_mem = devinfo_->devset->host_memory()->MakeBuffer(
      size_, hal::BufferAccessMask::HOST_READABLE | hal::BufferAccessMask::HOST_WRITEABLE);
  void* data = host_mem->MapDiscard({}).get();
  return compat::make_unique<CopyMemView>(ctx, devinfo_, deps_, data, size_, dev_mem_, std::move(host_mem));
}

}  // namespace

CopyMemStrategy::CopyMemStrategy(const std::shared_ptr<DevInfo>& devinfo) : devinfo_{devinfo} {
  if (!devinfo_->devset->host_memory() || !devinfo_->dev->executor() || !devinfo_->dev->executor()->device_memory()) {
    throw std::logic_error{"The copying memory management strategy requires both host and device memory"};
  }
}

std::shared_ptr<MemChunk> CopyMemStrategy::MakeChunk(const context::Context& ctx, std::uint64_t size) const {
  return std::make_shared<CopyMemChunk>(ctx, devinfo_, size);
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
