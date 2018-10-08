// Copyright 2018, Intel Corporation.

#include "tile/hal/cuda/hal.h"

#include "base/util/error.h"
#include "base/util/factory.h"
#include "tile/hal/cuda/error.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cuda {

[[gnu::unused]] char reg = []() -> char {
  FactoryRegistrar<hal::Driver>::Instance()->Register(
      "cuda",                                                                        //
      [](const context::Context& ctx) { return compat::make_unique<Driver>(ctx); },  //
      FactoryPriority::LOW);
  return 0;
}();

Driver::Driver(const context::Context& ctx) {  //
  device_sets_.emplace_back(std::make_shared<DeviceSet>());
}

DeviceSet::DeviceSet()  //
    : host_memory_{new HostMemory} {
  VLOG(1) << "Enumerating CUDA devices";

  Error err = cuInit(0);
  Error::Check(err, "cuInit(0) failed");

  int device_count;
  err = cuDeviceGetCount(&device_count);
  Error::Check(err, "cuDeviceGetCount() failed");

  for (int i = 0; i < device_count; i++) {
    devices_.emplace_back(new Device(i));
  }
}

std::shared_ptr<hal::Buffer> HostMemory::MakeBuffer(std::uint64_t size, BufferAccessMask access) {
  return std::make_shared<HostBuffer>(size);
}

std::shared_ptr<hal::Arena> HostMemory::MakeArena(std::uint64_t size, BufferAccessMask access) {
  throw error::Unimplemented("Not implemented: HostMemory::MakeArena");
}

std::shared_ptr<HostBuffer> HostBuffer::Downcast(const std::shared_ptr<hal::Buffer>& buffer) {
  auto buf = std::dynamic_pointer_cast<HostBuffer>(buffer);
  if (!buf) {
    throw error::InvalidArgument{"Incompatible buffer for Tile device"};
  }
  return buf;
}

HostBuffer::HostBuffer(std::uint64_t size) {  //
  buf_.resize(size, '\0');
}

boost::future<void*> HostBuffer::MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  void* ptr = buf_.data();
  return boost::make_ready_future(ptr);
}

boost::future<void*> HostBuffer::MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  void* ptr = buf_.data();
  return boost::make_ready_future(ptr);
}

std::shared_ptr<hal::Event> HostBuffer::Unmap(const context::Context& ctx) {
  auto now = std::chrono::high_resolution_clock::now();
  std::shared_ptr<hal::Result> result = std::make_shared<Result>(ctx, "tile::hal::cuda::HostBuffer::Unmap", now, now);
  return std::make_shared<Event>(std::move(result));
}

Device::Device(int didx) {
  Error err = cuDeviceGet(&device_, didx);
  Error::Check(err, "cuDeviceGet() failed");

  err = cuCtxCreate(&context_, 0, device_);
  Error::Check(err, "cuCtxCreate() failed");

  compiler_.reset(new Compiler(this));
  executor_.reset(new Executor(this));
}

Device::~Device() {
  if (compiler_) {
    Error err = cuCtxDestroy(context_);
    if (err) {
      LOG(ERROR) << "cuCtxDestroy() failed: " << err.str();
    }
  }
}

std::string Device::description() {
  char name[128];
  Error err = cuDeviceGetName(name, sizeof(name), device_);
  Error::Check(err, "cuDeviceGetName() failed");
  return printstring("NVIDIA %s (CUDA)", name);
}

void Device::SetCurrentContext() {
  Error err = cuCtxSetCurrent(context_);
  Error::Check(err, "cuCtxSetCurrent() failed");
}

hal::proto::HardwareInfo Device::GetHardwareInfo() {
  hal::proto::HardwareInfo info;

  char name[128];
  Error err = cuDeviceGetName(name, sizeof(name), device_);
  Error::Check(err, "cuDeviceGetName() failed");

  info.set_name(std::string("CUDA NVIDIA ") + name);
  info.set_vendor("NVIDIA");

  hal::proto::HardwareSettings* settings = info.mutable_settings();
  settings->set_threads(1);
  settings->set_vec_size(1);
  settings->set_use_global(false);
  settings->set_mem_width(64);
  settings->set_max_mem(32768);
  settings->set_max_regs(32);
  settings->set_goal_groups(1);
  settings->set_goal_flops_per_byte(1);
  settings->add_dim_sizes(0);

  return info;
}

DeviceMemory::DeviceMemory(Device* device)  //
    : device_(device)                       //
{}

std::shared_ptr<hal::Buffer> DeviceMemory::MakeBuffer(std::uint64_t size, BufferAccessMask access) {
  device_->SetCurrentContext();
  CUdeviceptr dptr;
  Error err = cuMemAlloc(&dptr, size);
  Error::Check(err, "cuMemAlloc() failed");
  return std::make_shared<DeviceBuffer>(device_, dptr, size);
}

std::shared_ptr<hal::Arena> DeviceMemory::MakeArena(std::uint64_t size, BufferAccessMask access) {
  throw error::Unimplemented("Not implemented: DeviceMemory::MakeArena");
}

std::shared_ptr<DeviceBuffer> DeviceBuffer::Downcast(const std::shared_ptr<hal::Buffer>& buffer) {
  auto buf = std::dynamic_pointer_cast<DeviceBuffer>(buffer);
  if (!buf) {
    throw error::InvalidArgument{"Incompatible buffer for Tile device"};
  }
  return buf;
}

DeviceBuffer::DeviceBuffer(Device* device, CUdeviceptr dptr, std::uint64_t size)
    : size_(size),     //
      dptr_(dptr),     //
      device_(device)  //
{}

DeviceBuffer::~DeviceBuffer() {
  device_->SetCurrentContext();
  cuMemFree(dptr_);
}

boost::future<void*> DeviceBuffer::MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  buf_.resize(size_);
  device_->SetCurrentContext();
  Error err = cuMemcpyDtoH(buf_.data(), dptr_, size_);
  Error::Check(err, "cuMemcpyDtoH() failed");
  void* ptr = buf_.data();
  return boost::make_ready_future(ptr);
}

boost::future<void*> DeviceBuffer::MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  buf_.resize(size_);
  void* ptr = buf_.data();
  return boost::make_ready_future(ptr);
}

std::shared_ptr<hal::Event> DeviceBuffer::Unmap(const context::Context& ctx) {
  device_->SetCurrentContext();
  auto start = std::chrono::high_resolution_clock::now();
  Error err = cuMemcpyHtoD(dptr_, buf_.data(), size_);
  Error::Check(err, "cuMemcpyHtoD() failed");
  auto end = std::chrono::high_resolution_clock::now();
  std::shared_ptr<hal::Result> result =
      std::make_shared<Result>(ctx, "tile::hal::cuda::DeviceBuffer::Unmap", start, end);
  return std::make_shared<Event>(std::move(result));
}

Executor::Executor(Device* device)              //
    : info_{device->GetHardwareInfo()},         //
      device_memory_(new DeviceMemory(device))  //
{}

std::shared_ptr<hal::Event> Executor::Copy(const context::Context& ctx,               //
                                           const std::shared_ptr<hal::Buffer>& from,  //
                                           std::size_t from_offset,                   //
                                           const std::shared_ptr<hal::Buffer>& to,    //
                                           std::size_t to_offset,                     //
                                           std::size_t length,                        //
                                           const std::vector<std::shared_ptr<hal::Event>>& dependencies) {
  throw error::Unimplemented("Not implemented: Executor::Copy");
}

boost::future<std::unique_ptr<hal::Executable>> Executor::Prepare(hal::Library* library, std::size_t kidx) {
  auto lib = Library::Downcast(library);
  return lib->Prepare(kidx);
}

boost::future<std::vector<std::shared_ptr<hal::Result>>> Executor::WaitFor(
    const std::vector<std::shared_ptr<hal::Event>>& events) {
  return Event::WaitFor(events);
}

void Executor::Flush() {}

Executable::Executable(std::vector<std::shared_ptr<Kernel>> kernels) : kernels_{std::move(kernels)} {}

std::shared_ptr<hal::Event> Executable::Run(const context::Context& ctx, std::size_t kernel_index,
                                            const std::vector<std::shared_ptr<Buffer>>& params,
                                            const std::vector<std::shared_ptr<Event>>& dependencies,
                                            bool enable_profiling = false) {
  return kernels_[kernel_index]->Run(ctx, params, dependencies, enable_profiling);
}

ComputeKernel::ComputeKernel(Device* device, const lang::KernelInfo& ki, CUfunction function)
    : ki_(ki),              //
      function_(function),  //
      device_(device)       //
{}

std::shared_ptr<hal::Event> ComputeKernel::Run(const context::Context& ctx,                              //
                                               const std::vector<std::shared_ptr<hal::Buffer>>& params,  //
                                               const std::vector<std::shared_ptr<hal::Event>>& deps,     //
                                               bool enable_profiling) {
  size_t shared_bytes = 0;

  lang::GridSize block{{
      ki_.lwork[0] ? ki_.lwork[0] : 1,  //
      ki_.lwork[1] ? ki_.lwork[1] : 1,  //
      ki_.lwork[2] ? ki_.lwork[2] : 1   //
  }};
  lang::GridSize grid{{
      ki_.gwork[0] / block[0],  //
      ki_.gwork[1] / block[1],  //
      ki_.gwork[2] / block[2]   //
  }};

  std::vector<void*> args;
  for (const auto& param : params) {
    auto buf = DeviceBuffer::Downcast(param);
    args.push_back(buf->dptr_arg());
  }

  device_->SetCurrentContext();

  auto start = std::chrono::high_resolution_clock::now();
  Error err = cuLaunchKernel(function_,     // f
                             grid[0],       // gridDimX
                             grid[1],       // gridDimY
                             grid[2],       // gridDimZ
                             block[0],      // blockDimX
                             block[1],      // blockDimY
                             block[2],      // blockDimZ
                             shared_bytes,  // sharedMemBytes
                             nullptr,       // hStream
                             args.data(),   // kernelParams
                             nullptr);      // extra
  Error::Check(err, "cuLaunchKernel() failed");
  err = cuCtxSynchronize();
  Error::Check(err, "cuCtxSynchronize() failed");
  auto end = std::chrono::high_resolution_clock::now();
  std::shared_ptr<hal::Result> result = std::make_shared<Result>(ctx, "tile::hal::cuda::Kernel::Run", start, end);
  return std::make_shared<Event>(std::move(result));
}

ZeroKernel::ZeroKernel(Device* device, const lang::KernelInfo& ki)  //
    : ki_(ki),                                                      //
      device_(device)                                               //
{}

std::shared_ptr<hal::Event> ZeroKernel::Run(const context::Context& ctx,
                                            const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                            const std::vector<std::shared_ptr<hal::Event>>& deps,
                                            bool enable_profiling) {
  auto buf = DeviceBuffer::Downcast(params[0]);
  auto dptr = buf->dptr();

  device_->SetCurrentContext();

  auto start = std::chrono::high_resolution_clock::now();

  Error err = cuMemsetD8(dptr, 0, buf->size());
  Error::Check(err, "cuMemsetD8() failed");

  err = cuCtxSynchronize();
  Error::Check(err, "cuCtxSynchronize() failed");

  auto end = std::chrono::high_resolution_clock::now();
  std::shared_ptr<hal::Result> result = std::make_shared<Result>(ctx, "tile::hal::cuda::ZeroKernel::Run", start, end);
  return std::make_shared<Event>(std::move(result));
}

Library::Library(std::vector<std::shared_ptr<Kernel>> kernels) : kernels_{std::move(kernels)} {}

Library* Library::Downcast(hal::Library* library) {  //
  return dynamic_cast<Library*>(library);
}

boost::future<std::unique_ptr<hal::Executable>> Library::Prepare() {
  return boost::make_ready_future<std::unique_ptr<hal::Executable>>(compat::make_unique<Executable>(kernels_));
}

Event::Event(std::shared_ptr<hal::Result> result)  //
    : result_(std::move(result))                   //
{}

std::shared_ptr<Event> Event::Downcast(const std::shared_ptr<hal::Event>& event) {
  auto evt = std::dynamic_pointer_cast<Event>(event);
  if (!evt) {
    throw error::InvalidArgument{"Incompatible event for Tile device"};
  }
  return evt;
}

boost::future<std::vector<std::shared_ptr<hal::Result>>> Event::WaitFor(
    const std::vector<std::shared_ptr<hal::Event>>& events) {
  std::vector<boost::shared_future<std::shared_ptr<hal::Result>>> futures;
  for (auto& event : events) {
    futures.emplace_back(event->GetFuture());
  }
  auto deps = boost::when_all(futures.begin(), futures.end());
  auto results = deps.then([](decltype(deps) fut) {
    std::vector<std::shared_ptr<hal::Result>> results;
    for (const auto& result : fut.get()) {
      results.emplace_back(result.get());
    }
    return results;
  });
  return results;
}

boost::shared_future<std::shared_ptr<hal::Result>> Event::GetFuture() {  //
  return boost::make_ready_future(result_);
}

Result::Result(const context::Context& ctx,                           //
               const char* verb,                                      //
               std::chrono::high_resolution_clock::time_point start,  //
               std::chrono::high_resolution_clock::time_point end)    //
    : ctx_{ctx},                                                      //
      verb_{verb},                                                    //
      start_{start},                                                  //
      end_{end}                                                       //
{}

std::chrono::high_resolution_clock::duration Result::GetDuration() const {  //
  return end_ - start_;
}

namespace {
const context::Clock kSystemClock;
}  // namespace

void Result::LogStatistics() const {
  google::protobuf::Duration start;
  google::protobuf::Duration end;
  context::StdDurationToProto(&start, start_.time_since_epoch());
  context::StdDurationToProto(&end, end_.time_since_epoch());
  kSystemClock.LogActivity(ctx_, verb_, start, end);
}

}  // namespace cuda
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
