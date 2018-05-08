// Copyright 2018, Vertex.AI.

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
  auto devices = mtlpp::Device::CopyAllDevices();
  for (size_t i = 0; i < devices.GetSize(); i++) {
    devices_.push_back(std::make_shared<Device>(devices[i]));
  }
}

Device::Device(mtlpp::Device device)  //
    : device_(device) {
  compiler_.reset(new Compiler(this));
  executor_.reset(new Executor(this));
}

std::string Device::description() {  //
  return device_.GetName().GetCStr();
}

hal::proto::HardwareInfo Device::GetHardwareInfo() {
  hal::proto::HardwareInfo info;

  info.set_name(std::string("Metal ") + description());
  info.set_vendor("Metal");

  hal::proto::HardwareSettings* settings = info.mutable_settings();
  settings->set_threads(1);
  settings->set_vec_size(1);
  settings->set_use_global(false);
  settings->set_mem_width(64);
  settings->set_max_mem(device_.GetRecommendedMaxWorkingSetSize());
  settings->set_max_regs(16 * 1024);
  settings->set_goal_groups(4);
  settings->set_goal_flops_per_byte(50);

  auto dims = device_.GetMaxThreadsPerThreadgroup();
  settings->add_dim_sizes(dims.Width);
  settings->add_dim_sizes(dims.Height);
  settings->add_dim_sizes(dims.Depth);

  return info;
}

mtlpp::CommandQueue Device::queue() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (!queue_) {
    queue_ = device_.NewCommandQueue();
  }
  return queue_;
}

Memory::Memory(Device* device)  //
    : device_(device)           //
{}

std::shared_ptr<hal::Buffer> Memory::MakeBuffer(std::uint64_t size, BufferAccessMask access) {
  mtlpp::ResourceOptions options;
  if ((access & BufferAccessMask::LOCATION_MASK) == BufferAccessMask::DEVICE) {
    // This is a device-only accessible buffer.
    options = mtlpp::ResourceOptions::StorageModePrivate;
  } else {
    options = mtlpp::ResourceOptions::StorageModeManaged;
  }
  auto buf = device_->dev().NewBuffer(size, options);
  return std::make_shared<Buffer>(device_, buf, size, access);
}

std::shared_ptr<hal::Arena> Memory::MakeArena(std::uint64_t size, BufferAccessMask access) {
  throw error::Unimplemented("Not implemented: Memory::MakeArena");
}

Buffer::Buffer(Device* device,           //
               mtlpp::Buffer buffer,     //
               std::uint64_t size,       //
               BufferAccessMask access)  //
    : device_(device),                   //
      buffer_(buffer),                   //
      size_(size),                       //
      access_(access)                    //
{}

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
  auto cmdbuf = device_->queue().CommandBuffer();
  auto encoder = cmdbuf.BlitCommandEncoder();
  encoder.Synchronize(buffer_);
  encoder.EndEncoding();
  auto promise = std::make_shared<boost::promise<void*>>();
  cmdbuf.AddCompletedHandler([this, promise](mtlpp::CommandBuffer cmdbuf) {  //
    promise->set_value(buffer_.GetContents());
  });
  cmdbuf.Commit();
  return promise->get_future();
}

boost::future<void*> Buffer::MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  if ((access_ & BufferAccessMask::LOCATION_MASK) == BufferAccessMask::DEVICE) {
    throw error::Unimplemented("Not Implemented: Buffer::MapDiscard for device-only accessible memory");
  }
  return boost::make_ready_future(buffer_.GetContents());
}

std::shared_ptr<hal::Event> Buffer::Unmap(const context::Context& ctx) {
  if ((access_ & BufferAccessMask::LOCATION_MASK) == BufferAccessMask::DEVICE) {
    throw error::Unimplemented("Not Implemented: Buffer::Unmap for device-only accessible memory");
  }
  buffer_.DidModify(ns::Range(0, size_));
  auto cmdbuf = device_->queue().CommandBuffer();
  auto encoder = cmdbuf.BlitCommandEncoder();
  encoder.EndEncoding();
  auto event = std::make_shared<Event>(ctx, cmdbuf, "tile::hal::opencl::Buffer::Unmap");
  cmdbuf.Commit();
  return event;
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

struct BuildResult {
  mtlpp::Library lib;
  ns::Error err;
};

}  // namespace

boost::future<std::unique_ptr<hal::Library>> Compiler::Build(const context::Context& ctx,
                                                             const std::vector<lang::KernelInfo>& kernel_infos,
                                                             const hal::proto::HardwareSettings& settings) {
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

  mtlpp::CompileOptions options;
  options.SetFastMathEnabled(true);
  auto promise = std::make_shared<boost::promise<std::unique_ptr<hal::Library>>>();
  device_->dev().NewLibrary(code.c_str(), options, [
    activity,          //
    device = device_,  //
    promise,           //
    kernel_ctxs = std::move(kernel_ctxs)
  ](const mtlpp::Library& lib, const ns::Error& err) mutable {  //
    try {
      if (activity->ctx().is_logging_events()) {
        opencl::proto::BuildInfo binfo;
        activity->AddMetadata(binfo);
      }
      std::string desc;
      if (err) {
        desc = err.GetLocalizedDescription().GetCStr();
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
  });
  return promise->get_future();
}

Library::Library(const context::Context& ctx,                    //
                 Device* device,                                 //
                 mtlpp::Library library,                         //
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

boost::future<std::unique_ptr<hal::Kernel>> Library::Prepare(std::size_t kidx) {
  context::Activity kbuild{ctx_, "tile::hal::opencl::BuildKernel"};
  const auto& kctx = kernel_ctxs_[kidx];

  opencl::proto::KernelInfo kinfo;
  kinfo.set_kname(kctx.ki.kname);

  *(kinfo.mutable_kinfo()) = kctx.ki.info;
  kbuild.AddMetadata(kinfo);

  auto kernel_id = kbuild.ctx().activity_id();

  if (kctx.ki.ktype == lang::KernelType::kZero) {
    // kinfo.set_src("// Builtin zero kernel");
    std::unique_ptr<hal::Kernel> kernel(new ZeroKernel(device_, kctx.ki, kernel_id));
    return boost::make_ready_future(std::move(kernel));
  }
  if (kctx.ki.ktype == lang::KernelType::kCopy) {
    std::unique_ptr<hal::Kernel> kernel(new CopyKernel(device_, kctx.ki, kernel_id));
    return boost::make_ready_future(std::move(kernel));
  }
  // kinfo.set_src(src);
  auto function = library_.NewFunction(kctx.ki.kname.c_str());
  auto promise = std::make_shared<boost::promise<std::unique_ptr<hal::Kernel>>>();
  device_->dev().NewComputePipelineState(function, [ promise, device = device_, kctx, kernel_id ](
                                                       const mtlpp::ComputePipelineState& state, const ns::Error& err) {
    try {
      if (err) {
        auto desc = err.GetLocalizedDescription().GetCStr();
        LOG(ERROR) << "NewComputePipelineState(" << kctx.ki.kname << ") failed: " << desc;
        LOG(ERROR) << "Source code: \n" << kctx.ki.comments << "\n" << kctx.src;
        throw std::runtime_error(desc);
      }
      std::unique_ptr<hal::Kernel> kernel(new Kernel(device, kctx.ki, kernel_id, state));
      promise->set_value(std::move(kernel));
    } catch (...) {
      try {
        promise->set_exception(std::current_exception());
      } catch (...) {
      }  // set_exception() may throw too
    }
  });
  return promise->get_future();
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

boost::future<std::unique_ptr<hal::Kernel>> Executor::Prepare(hal::Library* library, std::size_t kidx) {
  auto lib = Library::Downcast(library);
  return lib->Prepare(kidx);
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

Kernel::Kernel(Device* device,                        //
               const lang::KernelInfo& ki,            //
               context::proto::ActivityID kernel_id,  //
               mtlpp::ComputePipelineState state)     //
    : device_(device),                                //
      ki_(ki),                                        //
      kernel_id_(kernel_id),                          //
      state_(state)                                   //
{}

std::shared_ptr<hal::Event> Kernel::Run(const context::Context& ctx,                              //
                                        const std::vector<std::shared_ptr<hal::Buffer>>& params,  //
                                        const std::vector<std::shared_ptr<hal::Event>>& deps,     //
                                        bool enable_profiling) {
  mtlpp::Size threads(ki_.lwork[0], ki_.lwork[1], ki_.lwork[2]);
  mtlpp::Size groups(ki_.gwork[0] / threads.Width,   //
                     ki_.gwork[1] / threads.Height,  //
                     ki_.gwork[2] / threads.Depth);
  auto cmdbuf = device_->queue().CommandBuffer();
  auto encoder = cmdbuf.ComputeCommandEncoder();
  encoder.SetComputePipelineState(state_);
  for (size_t i = 0; i < params.size(); i++) {
    auto buf = Buffer::Downcast(params[i]);
    encoder.SetBuffer(buf->buffer(), 0, i);
  }
  encoder.DispatchThreadgroups(groups, threads);
  encoder.EndEncoding();
  context::Activity activity{ctx, "tile::hal::opencl::Kernel::Run"};
  if (ctx.is_logging_events()) {
    opencl::proto::RunInfo rinfo;
    *rinfo.mutable_kernel_id() = kernel_id_;
    activity.AddMetadata(rinfo);
  }
  auto event = std::make_shared<Event>(activity.ctx(), cmdbuf, "tile::hal::opencl::Executing");
  cmdbuf.Commit();
  return event;
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
  auto cmdbuf = device_->queue().CommandBuffer();
  auto encoder = cmdbuf.BlitCommandEncoder();
  auto dst_buf = Buffer::Downcast(params[0]);
  auto src_buf = Buffer::Downcast(params[1]);
  encoder.Copy(src_buf->buffer(), 0, dst_buf->buffer(), 0, src_buf->size());
  encoder.EndEncoding();
  context::Activity activity{ctx, "tile::hal::opencl::Buffer::Copy"};
  if (ctx.is_logging_events()) {
    opencl::proto::RunInfo rinfo;
    *rinfo.mutable_kernel_id() = kernel_id_;
    activity.AddMetadata(rinfo);
  }
  auto event = std::make_shared<Event>(activity.ctx(), cmdbuf, "tile::hal::opencl::Executing");
  cmdbuf.Commit();
  return event;
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
  auto cmdbuf = device_->queue().CommandBuffer();
  auto encoder = cmdbuf.BlitCommandEncoder();
  auto buf = Buffer::Downcast(params[0]);
  encoder.Fill(buf->buffer(), ns::Range(0, buf->size()), 0);
  encoder.EndEncoding();
  context::Activity activity{ctx, "tile::hal::opencl::Buffer::Fill"};
  if (ctx.is_logging_events()) {
    opencl::proto::RunInfo rinfo;
    *rinfo.mutable_kernel_id() = kernel_id_;
    activity.AddMetadata(rinfo);
  }
  auto event = std::make_shared<Event>(activity.ctx(), cmdbuf, "tile::hal::opencl::Executing");
  cmdbuf.Commit();
  return event;
}

Event::Event(const context::Context& ctx,  //
             mtlpp::CommandBuffer cmdbuf,  //
             const char* verb)             //
    : ctx_(ctx),                           //
      verb_(verb) {                        //
  auto promise = std::make_shared<boost::promise<std::shared_ptr<hal::Result>>>();
  future_ = promise->get_future();
  auto start = std::chrono::high_resolution_clock::now();
  cmdbuf.AddCompletedHandler([ ctx = ctx_, verb = verb_, start, promise ](mtlpp::CommandBuffer cmdbuf) {  //
    try {
      if (cmdbuf.GetStatus() == mtlpp::CommandBufferStatus::Error) {
        auto msg = cmdbuf.GetError().GetLocalizedDescription().GetCStr();
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
  });
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
