// Copyright 2018, Vertex.AI.

#pragma once

#include <string>
#include <vector>

#include "mtlpp.hpp"
#include "tile/base/hal.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace metal {

class Driver final : public hal::Driver {
 public:
  explicit Driver(const context::Context& ctx);

  const std::vector<std::shared_ptr<hal::DeviceSet>>& device_sets() final { return device_sets_; }

 private:
  std::vector<std::shared_ptr<hal::DeviceSet>> device_sets_;
};

class DeviceSet final : public hal::DeviceSet {
 public:
  DeviceSet();

  const std::vector<std::shared_ptr<hal::Device>>& devices() final { return devices_; }

  hal::Memory* host_memory() final { return nullptr; }

 private:
  std::vector<std::shared_ptr<hal::Device>> devices_;
};

class Device final : public hal::Device {
 public:
  explicit Device(mtlpp::Device device);

  void Initialize(const hal::proto::HardwareSettings& settings) final {
    // NOP
  }

  std::string description() final;

  hal::Compiler* compiler() final { return compiler_.get(); }

  hal::Loader* loader() final { return nullptr; }

  const std::unordered_map<std::string, std::unique_ptr<hal::Loader>>& il_loader_map() final { return il_loader_map_; }

  hal::Executor* executor() final { return executor_.get(); }

 public:
  hal::proto::HardwareInfo GetHardwareInfo();

  mtlpp::Device dev() const { return device_; }
  mtlpp::CommandQueue queue();

 private:
  mtlpp::Device device_;
  mtlpp::CommandQueue queue_;
  std::unique_ptr<hal::Compiler> compiler_;
  std::unique_ptr<hal::Executor> executor_;
  const std::unordered_map<std::string, std::unique_ptr<hal::Loader>> il_loader_map_;
  std::mutex mutex_;
};

class Memory final : public hal::Memory {
 public:
  explicit Memory(Device* device);

  std::uint64_t size_goal() const final { return 16 * std::giga::num; }
  BufferAccessMask AllowedAccesses() const final { return BufferAccessMask::ALL; }
  std::size_t ArenaBufferAlignment() const final { return alignof(long double); }

  std::shared_ptr<hal::Buffer> MakeBuffer(std::uint64_t size, BufferAccessMask access) final;
  std::shared_ptr<hal::Arena> MakeArena(std::uint64_t size, BufferAccessMask access) final;

 private:
  Device* device_;
};

class Buffer : public hal::Buffer {
 public:
  Buffer(Device* device,        //
         mtlpp::Buffer buffer,  //
         std::uint64_t size,    //
         BufferAccessMask access);

  boost::future<void*> MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  boost::future<void*> MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  std::shared_ptr<hal::Event> Unmap(const context::Context& ctx) final;

 public:
  static std::shared_ptr<Buffer> Downcast(const std::shared_ptr<hal::Buffer>& buffer);

  mtlpp::Buffer buffer() const { return buffer_; }
  std::uint64_t size() const { return size_; }

 private:
  Device* device_;
  mtlpp::Buffer buffer_;
  std::uint64_t size_;
  BufferAccessMask access_;
};

class Compiler final : public hal::Compiler {
 public:
  explicit Compiler(Device* device);

  boost::future<std::unique_ptr<hal::Library>> Build(const context::Context& ctx,
                                                     const std::vector<lang::KernelInfo>& kernel_infos,
                                                     const hal::proto::HardwareSettings& settings) final;

 private:
  Device* device_;
};

class Executor : public hal::Executor {
 public:
  explicit Executor(Device* device);

  const hal::proto::HardwareInfo& info() final { return info_; }

  hal::Memory* device_memory() final { return memory_.get(); }

  hal::Memory* shared_memory() final { return nullptr; }

  bool is_synchronous() const final { return true; }

  std::shared_ptr<hal::Event> Copy(const context::Context& ctx,               //
                                   const std::shared_ptr<hal::Buffer>& from,  //
                                   std::size_t from_offset,                   //
                                   const std::shared_ptr<hal::Buffer>& to,    //
                                   std::size_t to_offset,                     //
                                   std::size_t length,
                                   const std::vector<std::shared_ptr<hal::Event>>& dependencies) final;

  boost::future<std::unique_ptr<hal::Kernel>> Prepare(hal::Library* library, std::size_t kidx) final;

  boost::future<std::vector<std::shared_ptr<hal::Result>>> WaitFor(
      const std::vector<std::shared_ptr<hal::Event>>& events) final;

  void Flush() final {
    // NOP
  }

 private:
  const hal::proto::HardwareInfo info_;
  std::unique_ptr<hal::Memory> memory_;
};

class Kernel final : public hal::Kernel {
 public:
  Kernel(Device* device,                        //
         const lang::KernelInfo& ki,            //
         context::proto::ActivityID kernel_id,  //
         mtlpp::ComputePipelineState state);

  std::shared_ptr<hal::Event> Run(const context::Context& ctx,  //
                                  const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                  const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                  bool enable_profiling) final;

 private:
  Device* device_;
  lang::KernelInfo ki_;
  context::proto::ActivityID kernel_id_;
  mtlpp::ComputePipelineState state_;
};

class CopyKernel final : public hal::Kernel {
 public:
  CopyKernel(Device* device,              //
             const lang::KernelInfo& ki,  //
             context::proto::ActivityID kernel_id);

  std::shared_ptr<hal::Event> Run(const context::Context& ctx,  //
                                  const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                  const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                  bool enable_profiling) final;

 private:
  Device* device_;
  lang::KernelInfo ki_;
  context::proto::ActivityID kernel_id_;
};

class ZeroKernel final : public hal::Kernel {
 public:
  ZeroKernel(Device* device,              //
             const lang::KernelInfo& ki,  //
             context::proto::ActivityID kernel_id);

  std::shared_ptr<hal::Event> Run(const context::Context& ctx,  //
                                  const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                  const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                  bool enable_profiling) final;

 private:
  Device* device_;
  lang::KernelInfo ki_;
  context::proto::ActivityID kernel_id_;
};

struct KernelContext {
  lang::KernelInfo ki;
  std::string src;
};

std::string EmitMetal(const lang::KernelInfo& ki);

class Library final : public hal::Library {
 public:
  Library(const context::Context& ctx,  //
          Device* device,               //
          mtlpp::Library library,       //
          const std::vector<KernelContext>& kernel_ctxs);

  std::string Serialize() final;

 public:
  static Library* Downcast(hal::Library* library);

  boost::future<std::unique_ptr<hal::Kernel>> Prepare(std::size_t kidx);

 private:
  context::Context ctx_;
  Device* device_;
  mtlpp::Library library_;
  std::vector<KernelContext> kernel_ctxs_;
};

class Event final : public hal::Event {
 public:
  Event(const context::Context& ctx,  //
        mtlpp::CommandBuffer cmdbuf,  //
        const char* verb);

  boost::shared_future<std::shared_ptr<hal::Result>> GetFuture() final { return future_; }

 private:
  context::Context ctx_;
  const char* verb_;
  boost::shared_future<std::shared_ptr<hal::Result>> future_;
};

class Result final : public hal::Result {
 public:
  Result(const context::Context& ctx,                           //
         const char* verb,                                      //
         std::chrono::high_resolution_clock::time_point start,  //
         std::chrono::high_resolution_clock::time_point end);

  std::chrono::high_resolution_clock::duration GetDuration() const final;
  void LogStatistics() const final;

 private:
  context::Context ctx_;
  const char* verb_;
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::high_resolution_clock::time_point end_;
};

}  // namespace metal
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
