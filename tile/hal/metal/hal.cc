// Copyright 2018, Intel Corporation.

#include "tile/hal/metal/hal.h"

#include "base/util/error.h"
#include "base/util/factory.h"
#include "tile/hal/opencl/opencl.pb.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace metal {

[[gnu::unused]] char reg = []() -> char {
  FactoryRegistrar<hal::Driver>::Instance()->Register(
      "metal",                                                                       //
      [](const context::Context& ctx) { return compat::make_unique<Driver>(ctx); },  //
      FactoryPriority::HIGH);
  return 0;
}();

Driver::Driver(const context::Context& ctx) {  //
  device_sets_.emplace_back(std::make_shared<DeviceSet>());
}

DeviceSet::DeviceSet() {
  VLOG(1) << "Enumerating Metal devices";
  auto devices = MTLCopyAllDevices();
  for (size_t i = 0; i < [devices count]; i++) {
    devices_.push_back(std::make_shared<Device>([devices objectAtIndexedSubscript:i]));
  }
}

Device::Device(id<MTLDevice> device)  //
    : device_(device) {
  compiler_.reset(new Compiler(this));
  executor_.reset(new Executor(this));
}

std::string Device::name() {  //
  return [[device_ name] cStringUsingEncoding:NSUTF8StringEncoding];
}

std::string Device::description() {  //
  return name() + " (Metal)";
}

hal::proto::HardwareInfo Device::GetHardwareInfo() {
  hal::proto::HardwareInfo info;

  info.set_name(std::string("Metal ") + name());
  info.set_vendor("Metal");

  hal::proto::HardwareSettings* settings = info.mutable_settings();
  settings->set_threads(1);
  settings->set_vec_size(1);
  settings->set_use_global(false);
  settings->set_mem_width(64);
  settings->set_max_mem([device_ recommendedMaxWorkingSetSize]);
  settings->set_max_regs(16 * 1024);
  settings->set_goal_groups(4);
  settings->set_goal_flops_per_byte(50);

  auto dims = [device_ maxThreadsPerThreadgroup];
  settings->add_dim_sizes(dims.width);
  settings->add_dim_sizes(dims.height);
  settings->add_dim_sizes(dims.depth);

  return info;
}

id<MTLCommandQueue> Device::queue() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (!queue_) {
    queue_ = [device_ newCommandQueue];
  }
  return queue_;
}

Memory::Memory(Device* device)  //
    : device_(device)           //
{}

std::shared_ptr<hal::Buffer> Memory::MakeBuffer(std::uint64_t size, BufferAccessMask access) {
  MTLResourceOptions options;
  if ((access & BufferAccessMask::LOCATION_MASK) == BufferAccessMask::DEVICE) {
    // This is a device-only accessible buffer.
    options = MTLResourceStorageModePrivate;
  } else {
    options = MTLResourceStorageModeManaged;
  }
  auto buf = [device_->dev() newBufferWithLength:size options:options];
  return std::make_shared<Buffer>(device_, buf, size, access);
}

std::shared_ptr<hal::Arena> Memory::MakeArena(std::uint64_t size, BufferAccessMask access) {
  throw error::Unimplemented("Not implemented: Memory::MakeArena");
}

Buffer::Buffer(Device* device,           //
               id<MTLBuffer> buffer,     //
               std::uint64_t size,       //
               BufferAccessMask access)  //
    : device_(device),                   //
      buffer_(buffer),                   //
      size_(size),                       //
      access_(access) {                  //
}

std::shared_ptr<Buffer> Buffer::Downcast(const std::shared_ptr<hal::Buffer>& buffer) {
  auto buf = std::dynamic_pointer_cast<Buffer>(buffer);
  if (!buf) {
    throw error::InvalidArgument{"Incompatible buffer for Tile device"};
  }
  return buf;
}

boost::future<void*> Buffer::MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  if ((access_ & BufferAccessMask::LOCATION_MASK) == BufferAccessMask::DEVICE) {
    throw error::Unimplemented("Not Implemented: Buffer::MapCurrent for device-only accessible memory");
  }
  auto promise = std::make_shared<boost::promise<void*>>();
  auto handler = [buffer = buffer_, promise](id<MTLCommandBuffer> cmdbuf) mutable {
    promise->set_value([buffer contents]);
  };
  @autoreleasepool {
    auto cmdbuf = [device_->queue() commandBuffer];
    auto encoder = [cmdbuf blitCommandEncoder];
    [encoder synchronizeResource:buffer_];
    [encoder endEncoding];
    [cmdbuf addCompletedHandler:handler];
    [cmdbuf commit];
    return promise->get_future();
  }
}

boost::future<void*> Buffer::MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  if ((access_ & BufferAccessMask::LOCATION_MASK) == BufferAccessMask::DEVICE) {
    throw error::Unimplemented("Not Implemented: Buffer::MapDiscard for device-only accessible memory");
  }
  return boost::make_ready_future([buffer_ contents]);
}

std::shared_ptr<hal::Event> Buffer::Unmap(const context::Context& ctx) {
  if ((access_ & BufferAccessMask::LOCATION_MASK) == BufferAccessMask::DEVICE) {
    throw error::Unimplemented("Not Implemented: Buffer::Unmap for device-only accessible memory");
  }
  @autoreleasepool {
    [buffer_ didModifyRange:NSMakeRange(0, size_)];
    // NOTE: This encoder which seems to do nothing appears to be required for some Metal devices.
    auto cmdbuf = [device_->queue() commandBuffer];
    auto encoder = [cmdbuf blitCommandEncoder];
    [encoder endEncoding];
    auto event = std::make_shared<Event>(ctx, cmdbuf, "tile::hal::opencl::Buffer::Unmap");
    [cmdbuf commit];
    return event;
  }
}

Compiler::Compiler(Device* device)  //
    : device_(device)               //
{}

namespace {

std::string WithLineNumbers(const std::string& src) {
  std::stringstream ss_in(src);
  std::stringstream ss_out;
  size_t line_num = 1;
  std::string line;
  while (std::getline(ss_in, line, '\n')) {
    ss_out << std::setw(5) << line_num++ << ": " << line << "\n";
  }
  return ss_out.str();
}

}  // namespace

boost::future<std::unique_ptr<hal::Library>> Compiler::Build(const context::Context& ctx,
                                                             const std::vector<lang::KernelInfo>& kernel_infos,
                                                             const hal::proto::HardwareSettings& settings) {
  if (!kernel_infos.size()) {
    return boost::make_ready_future(std::unique_ptr<hal::Library>{
        compat::make_unique<Library>(ctx, device_, nullptr, std::vector<KernelContext>{})});
  }

  auto activity = std::make_shared<context::Activity>(ctx, "tile::hal::opencl::Build");

  std::vector<KernelContext> kernel_ctxs;
  std::map<std::string, std::string> seen;
  std::stringstream src;
  src << "using namespace metal;\n\n";
  for (const auto& ki : kernel_infos) {
    auto it = seen.find(ki.kname);
    if (it == seen.end()) {
      auto kernel_src = EmitMetal(ki);
      src << ki.comments;
      src << kernel_src;
      src << "\n\n";
      seen.insert(std::make_pair(ki.kname, kernel_src));
      kernel_ctxs.emplace_back(KernelContext{ki, kernel_src});
    } else {
      kernel_ctxs.emplace_back(KernelContext{ki, it->second});
    }
  }

  auto code = src.str();
  IVLOG(3, "Compiling Metal:\n" << WithLineNumbers(code));

  auto options = [[MTLCompileOptions alloc] init];
  options.fastMathEnabled = true;
  auto promise = std::make_shared<boost::promise<std::unique_ptr<hal::Library>>>();
  auto handler = [activity,          //
                  device = device_,  //
                  promise,           //
                  kernel_ctxs = std::move(kernel_ctxs)](id<MTLLibrary> lib, NSError* err) mutable {
    try {
      if (activity->ctx().is_logging_events()) {
        opencl::proto::BuildInfo binfo;
        activity->AddMetadata(binfo);
      }
      std::string desc;
      if (err) {
        desc = [[err localizedDescription] cStringUsingEncoding:NSUTF8StringEncoding];
        VLOG(1) << "Build log:\n" << desc;
      }
      if (!lib) {
        LOG(ERROR) << "Compiler::Build> Compilation failure:\n" << desc;
        throw std::runtime_error(desc);
      }
      std::unique_ptr<hal::Library> library(new Library(activity->ctx(), device, lib, kernel_ctxs));
      promise->set_value(std::move(library));
    } catch (...) {
      try {
        promise->set_exception(std::current_exception());
      } catch (...) {
      }  // set_exception() may throw too
    }
  };
  [device_->dev() newLibraryWithSource:[NSString stringWithUTF8String:code.c_str()]
                               options:options
                     completionHandler:handler];
  return promise->get_future();
}

Library::Library(const context::Context& ctx,                    //
                 Device* device,                                 //
                 id<MTLLibrary> library,                         //
                 const std::vector<KernelContext>& kernel_ctxs)  //
    : ctx_(ctx),                                                 //
      device_(device),                                           //
      library_(library),                                         //
      kernel_ctxs_(kernel_ctxs)                                  //
{}

Library* Library::Downcast(hal::Library* library) {  //
  return dynamic_cast<Library*>(library);
}

std::string Library::Serialize() {  //
  throw error::Unimplemented("Not implemented: Library::Serialize");
}

boost::future<std::unique_ptr<hal::Executable>> Library::Prepare() {
  std::vector<boost::future<std::unique_ptr<Kernel>>> kernels;
  kernels.reserve(kernel_ctxs_.size());

  for (std::size_t kidx = 0; kidx < kernel_ctxs_.size(); ++kidx) {
    context::Activity kbuild{ctx_, "tile::hal::opencl::BuildKernel"};
    const auto& kctx = kernel_ctxs_[kidx];

    opencl::proto::KernelInfo kinfo;
    kinfo.set_kname(kctx.ki.kname);

    *(kinfo.mutable_kinfo()) = kctx.ki.info;
    kbuild.AddMetadata(kinfo);

    auto kernel_id = kbuild.ctx().activity_id();

    if (kctx.ki.ktype == lang::KernelType::kZero) {
      // kinfo.set_src("// Builtin zero kernel");
      kernels.emplace_back(boost::make_ready_future(
          std::unique_ptr<Kernel>(compat::make_unique<ZeroKernel>(device_, kctx.ki, kernel_id))));
      continue;
    }
    if (kctx.ki.ktype == lang::KernelType::kCopy) {
      kernels.emplace_back(boost::make_ready_future(
          std::unique_ptr<Kernel>(compat::make_unique<CopyKernel>(device_, kctx.ki, kernel_id))));
      continue;
    }
    // kinfo.set_src(src);
    auto function = [library_ newFunctionWithName:[NSString stringWithUTF8String:kctx.ki.kname.c_str()]];
    auto promise = std::make_shared<boost::promise<std::unique_ptr<Kernel>>>();
    auto handler = [promise,           //
                    device = device_,  //
                    kctx,              //
                    kernel_id](const id<MTLComputePipelineState> state, NSError* err) {
      try {
        if (err) {
          auto desc = [[err localizedDescription] cStringUsingEncoding:NSUTF8StringEncoding];
          LOG(ERROR) << "NewComputePipelineState(" << kctx.ki.kname << ") failed: " << desc;
          LOG(ERROR) << "Source code: \n" << kctx.ki.comments << "\n" << kctx.src;
          throw std::runtime_error(desc);
        }
        std::unique_ptr<Kernel> kernel = compat::make_unique<ComputeKernel>(device, kctx.ki, kernel_id, state);
        promise->set_value(std::move(kernel));
      } catch (...) {
        try {
          promise->set_exception(std::current_exception());
        } catch (...) {
        }  // set_exception() may throw too
      }
    };
    [device_->dev() newComputePipelineStateWithFunction:function  //
                                      completionHandler:handler];
    kernels.emplace_back(promise->get_future());
  }

  return boost::when_all(kernels.begin(), kernels.end())
      .then([](boost::future<std::vector<boost::future<std::unique_ptr<Kernel>>>> kernel_futures_future) {
        std::vector<boost::future<std::unique_ptr<Kernel>>> kernel_futures = kernel_futures_future.get();
        std::vector<std::unique_ptr<Kernel>> kernels;
        kernels.reserve(kernel_futures.size());
        for (auto& fut : kernel_futures) {
          kernels.emplace_back(fut.get());
        }
        return std::unique_ptr<hal::Executable>(compat::make_unique<Executable>(std::move(kernels)));
      });
}

Executor::Executor(Device* device)       //
    : info_(device->GetHardwareInfo()),  //
      memory_(new Memory(device))        //
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

boost::future<std::unique_ptr<hal::Executable>> Executor::Prepare(hal::Library* library) {
  auto lib = Library::Downcast(library);
  return lib->Prepare();
}

boost::future<std::vector<std::shared_ptr<hal::Result>>> Executor::WaitFor(
    const std::vector<std::shared_ptr<hal::Event>>& events) {
  std::vector<boost::shared_future<std::shared_ptr<hal::Result>>> futures;
  for (auto& event : events) {
    futures.emplace_back(event->GetFuture());
  }
  auto all_futures = boost::when_all(futures.begin(), futures.end());
  auto results_future = all_futures.then([](decltype(all_futures) future) {
    std::vector<std::shared_ptr<hal::Result>> results;
    for (const auto& result : future.get()) {
      results.emplace_back(result.get());
    }
    return results;
  });
  return results_future;
}

Executable::Executable(std::vector<std::unique_ptr<Kernel>> kernels) : kernels_{std::move(kernels)} {}

std::shared_ptr<hal::Event> Executable::Run(const context::Context& ctx, std::size_t kernel_index,
                                            const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                            const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                            bool enable_profiling) {
  return kernels_[kernel_index]->Run(ctx, params, dependencies, enable_profiling);
}

ComputeKernel::ComputeKernel(Device* device, const lang::KernelInfo& ki, context::proto::ActivityID kernel_id,
                             id<MTLComputePipelineState> state)
    : device_(device), ki_(ki), kernel_id_(kernel_id), state_(state) {}

std::shared_ptr<hal::Event> ComputeKernel::Run(const context::Context& ctx,
                                               const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                               const std::vector<std::shared_ptr<hal::Event>>& deps,
                                               bool enable_profiling) {
  @autoreleasepool {
    MTLSize threads = MTLSizeMake(ki_.lwork[0], ki_.lwork[1], ki_.lwork[2]);
    MTLSize groups = MTLSizeMake(ki_.gwork[0] / threads.width,   //
                                 ki_.gwork[1] / threads.height,  //
                                 ki_.gwork[2] / threads.depth);
    auto cmdbuf = [device_->queue() commandBuffer];
    auto encoder = [cmdbuf computeCommandEncoder];
    [encoder setComputePipelineState:state_];
    for (size_t i = 0; i < params.size(); i++) {
      auto buf = Buffer::Downcast(params[i]);
      [encoder setBuffer:buf->buffer()  //
                  offset:0
                 atIndex:i];
    }
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [encoder endEncoding];
    context::Activity activity{ctx, "tile::hal::opencl::Kernel::Run"};
    if (ctx.is_logging_events()) {
      opencl::proto::RunInfo rinfo;
      *rinfo.mutable_kernel_id() = kernel_id_;
      activity.AddMetadata(rinfo);
    }
    auto event = std::make_shared<Event>(activity.ctx(), cmdbuf, "tile::hal::opencl::Executing");
    [cmdbuf commit];
    return event;
  }
}

CopyKernel::CopyKernel(Device* device,                        //
                       const lang::KernelInfo& ki,            //
                       context::proto::ActivityID kernel_id)  //
    : device_(device),                                        //
      ki_(ki),                                                //
      kernel_id_(kernel_id)                                   //
{}

std::shared_ptr<hal::Event> CopyKernel::Run(const context::Context& ctx,
                                            const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                            const std::vector<std::shared_ptr<hal::Event>>& deps,
                                            bool enable_profiling) {
  @autoreleasepool {
    auto cmdbuf = [device_->queue() commandBuffer];
    auto encoder = [cmdbuf blitCommandEncoder];
    auto dst_buf = Buffer::Downcast(params[0]);
    auto src_buf = Buffer::Downcast(params[1]);
    [encoder copyFromBuffer:src_buf->buffer()
               sourceOffset:0
                   toBuffer:dst_buf->buffer()
          destinationOffset:0
                       size:src_buf->size()];
    [encoder endEncoding];
    context::Activity activity{ctx, "tile::hal::opencl::Buffer::Copy"};
    if (ctx.is_logging_events()) {
      opencl::proto::RunInfo rinfo;
      *rinfo.mutable_kernel_id() = kernel_id_;
      activity.AddMetadata(rinfo);
    }
    auto event = std::make_shared<Event>(activity.ctx(), cmdbuf, "tile::hal::opencl::Executing");
    [cmdbuf commit];
    return event;
  }
}

ZeroKernel::ZeroKernel(Device* device,                        //
                       const lang::KernelInfo& ki,            //
                       context::proto::ActivityID kernel_id)  //
    : device_(device),                                        //
      ki_(ki),                                                //
      kernel_id_(kernel_id)                                   //
{}

std::shared_ptr<hal::Event> ZeroKernel::Run(const context::Context& ctx,
                                            const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                            const std::vector<std::shared_ptr<hal::Event>>& deps,
                                            bool enable_profiling) {
  @autoreleasepool {
    auto cmdbuf = [device_->queue() commandBuffer];
    auto encoder = [cmdbuf blitCommandEncoder];
    auto buf = Buffer::Downcast(params[0]);
    [encoder fillBuffer:buf->buffer()  //
                  range:NSMakeRange(0, buf->size())
                  value:0];
    [encoder endEncoding];
    context::Activity activity{ctx, "tile::hal::opencl::Buffer::Fill"};
    if (ctx.is_logging_events()) {
      opencl::proto::RunInfo rinfo;
      *rinfo.mutable_kernel_id() = kernel_id_;
      activity.AddMetadata(rinfo);
    }
    auto event = std::make_shared<Event>(activity.ctx(), cmdbuf, "tile::hal::opencl::Executing");
    [cmdbuf commit];
    return event;
  }
}

Event::Event(const context::Context& ctx,  //
             id<MTLCommandBuffer> cmdbuf,  //
             const char* verb)             //
    : ctx_(ctx),                           //
      verb_(verb) {                        //
  auto promise = std::make_shared<boost::promise<std::shared_ptr<hal::Result>>>();
  future_ = promise->get_future();
  auto start = std::chrono::high_resolution_clock::now();
  auto handler = [ctx = ctx_,    //
                  verb = verb_,  //
                  start,         //
                  promise](id<MTLCommandBuffer> cmdbuf) {
    try {
      if ([cmdbuf status] == MTLCommandBufferStatusError) {
        auto msg = [[[cmdbuf error] localizedDescription] cStringUsingEncoding:NSUTF8StringEncoding];
        LOG(ERROR) << msg;
        throw std::runtime_error(std::string("Kernel execution failure: ") + msg);
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::shared_ptr<hal::Result> result = std::make_shared<Result>(ctx, verb, start, end);
      promise->set_value(result);
    } catch (...) {
      try {
        promise->set_exception(std::current_exception());
      } catch (...) {
      }  // set_exception() may throw too
    }
  };
  [cmdbuf addCompletedHandler:handler];
}

Event::Event(const context::Context& ctx,  //
             const char* verb)             //
    : ctx_(ctx),                           //
      verb_(verb) {                        //
  boost::promise<std::shared_ptr<hal::Result>> promise;
  future_ = promise.get_future();
  auto now = std::chrono::high_resolution_clock::now();
  std::shared_ptr<hal::Result> result = std::make_shared<Result>(ctx, verb, now, now);
  promise.set_value(result);
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

}  // namespace metal
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
