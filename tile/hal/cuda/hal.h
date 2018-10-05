// Copyright 2018, Intel Corporation.

#pragma once

#include <string>
#include <vector>

#include "cuda/include/cuda.h"
#include "tile/base/hal.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cuda {

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

  hal::Memory* host_memory() final { return host_memory_.get(); }

 private:
  std::vector<std::shared_ptr<hal::Device>> devices_;
  std::unique_ptr<hal::Memory> host_memory_;
};

class HostMemory final : public hal::Memory {
 public:
  std::uint64_t size_goal() const final { return 16 * std::giga::num; }
  BufferAccessMask AllowedAccesses() const final { return BufferAccessMask::ALL; }
  std::size_t ArenaBufferAlignment() const final { return alignof(long double); }

  std::shared_ptr<hal::Buffer> MakeBuffer(std::uint64_t size, BufferAccessMask access) final;
  std::shared_ptr<hal::Arena> MakeArena(std::uint64_t size, BufferAccessMask access) final;
};

class HostBuffer : public hal::Buffer {
 public:
  explicit HostBuffer(std::uint64_t size);

  boost::future<void*> MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  boost::future<void*> MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  std::shared_ptr<hal::Event> Unmap(const context::Context& ctx) final;

  static std::shared_ptr<HostBuffer> Downcast(const std::shared_ptr<hal::Buffer>& buffer);

 private:
  std::vector<char> buf_;
};

class Device final : public hal::Device {
 public:
  explicit Device(int didx);
  ~Device();

  void Initialize(const hal::proto::HardwareSettings& settings) final {
    // NOP
  }

  std::string description() final;

  hal::Compiler* compiler() final { return compiler_.get(); }

  hal::Loader* loader() final { return nullptr; }

  const std::unordered_map<std::string, std::unique_ptr<hal::Loader>>& il_loader_map() final { return il_loader_map_; }

  hal::Executor* executor() final { return executor_.get(); }

  void SetCurrentContext();
  hal::proto::HardwareInfo GetHardwareInfo();

 private:
  std::unique_ptr<hal::Compiler> compiler_;
  std::unique_ptr<hal::Executor> executor_;
  const std::unordered_map<std::string, std::unique_ptr<hal::Loader>> il_loader_map_;
  CUdevice device_;
  CUcontext context_;
};

class DeviceMemory final : public hal::Memory {
 public:
  explicit DeviceMemory(Device* device);

  std::uint64_t size_goal() const final { return 16 * std::giga::num; }
  BufferAccessMask AllowedAccesses() const final { return BufferAccessMask::ALL; }
  std::size_t ArenaBufferAlignment() const final { return alignof(long double); }

  std::shared_ptr<hal::Buffer> MakeBuffer(std::uint64_t size, BufferAccessMask access) final;
  std::shared_ptr<hal::Arena> MakeArena(std::uint64_t size, BufferAccessMask access) final;

 private:
  Device* device_;
};

class DeviceBuffer : public hal::Buffer {
 public:
  DeviceBuffer(Device* device, CUdeviceptr dptr, std::uint64_t size);
  ~DeviceBuffer();

  boost::future<void*> MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  boost::future<void*> MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  std::shared_ptr<hal::Event> Unmap(const context::Context& ctx) final;

  static std::shared_ptr<DeviceBuffer> Downcast(const std::shared_ptr<hal::Buffer>& buffer);

  std::uint64_t size() const { return size_; }
  CUdeviceptr dptr() const { return dptr_; }
  void* dptr_arg() { return &dptr_; }

 private:
  std::uint64_t size_;
  std::vector<char> buf_;
  CUdeviceptr dptr_;
  Device* device_;
};

class Compiler final : public hal::Compiler {
 public:
  explicit Compiler(Device* device);

  boost::future<std::unique_ptr<hal::Library>> Build(const context::Context& ctx,
                                                     const std::vector<lang::KernelInfo>& kernels,
                                                     const hal::proto::HardwareSettings& /* settings */) final;

 private:
  Device* device_;
};

class Executor : public hal::Executor {
 public:
  explicit Executor(Device* device);

  const hal::proto::HardwareInfo& info() final { return info_; }

  Memory* device_memory() final { return device_memory_.get(); }

  Memory* shared_memory() final { return nullptr; }

  bool is_synchronous() const final { return true; }

  std::shared_ptr<hal::Event> Copy(const context::Context& ctx, const std::shared_ptr<hal::Buffer>& from,
                                   std::size_t from_offset, const std::shared_ptr<hal::Buffer>& to,
                                   std::size_t to_offset, std::size_t length,
                                   const std::vector<std::shared_ptr<hal::Event>>& dependencies) final;

  boost::future<std::unique_ptr<hal::Executable>> Prepare(hal::Library* library) final;

  boost::future<std::vector<std::shared_ptr<hal::Result>>> WaitFor(
      const std::vector<std::shared_ptr<hal::Event>>& events) final;

  void Flush() final;

 private:
  const hal::proto::HardwareInfo info_;
  std::unique_ptr<hal::Memory> device_memory_;
};

class Kernel {
 public:
  virtual ~Kernel() {}

  virtual std::shared_ptr<hal::Event> Run(const context::Context& ctx,
                                          const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                          const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                          bool enable_profiling) = 0;
};

class Executable final : public hal::Executable {
 public:
  explicit Executable(std::vector<std::unique_ptr<hal::Kernel>> kernels);

  std::shared_ptr<hal::Event> Run(const context::Context& ctx, std::size_t kernel_index,
                                  const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                  const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                  bool enable_profiling = false) final;

 private:
  std::vector<std::shared_ptr<Kernel>> kernels_;
};

class ComputeKernel final : public Kernel {
 public:
  ComputeKernel(Device* device, const lang::KernelInfo& ki, CUfunction function);

  std::shared_ptr<hal::Event> Run(const context::Context& ctx, const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                  const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                  bool enable_profiling) final;

 private:
  lang::KernelInfo ki_;
  CUfunction function_;
  Device* device_;
};

class ZeroKernel final : public Kernel {
 public:
  explicit ZeroKernel(Device* device, const lang::KernelInfo& ki);

  std::shared_ptr<hal::Event> Run(const context::Context& ctx, const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                  const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                  bool enable_profiling) final;

 private:
  lang::KernelInfo ki_;
  Device* device_;
};

class Library final : public hal::Library {
 public:
  static Library* Downcast(hal::Library* library);

  explicit Library(std::vector<std::shared_ptr<Kernel>> kernels);

  std::string Serialize() final { return ""; }

  boost::future<std::unique_ptr<hal::Executable>> Prepare();

 private:
  std::vector<std::shared_ptr<Kernel>> kernels_;
};

class Event final : public hal::Event {
 public:
  explicit Event(std::shared_ptr<hal::Result> result);

  static std::shared_ptr<Event> Downcast(const std::shared_ptr<hal::Event>& event);

  static boost::future<std::vector<std::shared_ptr<hal::Result>>> WaitFor(
      const std::vector<std::shared_ptr<hal::Event>>& events);

  boost::shared_future<std::shared_ptr<hal::Result>> GetFuture() final;

 private:
  std::shared_ptr<hal::Result> result_;
};

class Result final : public hal::Result {
 public:
  Result(const context::Context& ctx, const char* verb, std::chrono::high_resolution_clock::time_point start,
         std::chrono::high_resolution_clock::time_point end);

  std::chrono::high_resolution_clock::duration GetDuration() const final;
  void LogStatistics() const final;

 private:
  context::Context ctx_;
  const char* verb_;
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::high_resolution_clock::time_point end_;
};

}  // namespace cuda
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
