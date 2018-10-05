// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/opencl/compiler.h"

#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/filesystem.hpp>

#include "base/util/callback_map.h"
#include "base/util/compat.h"
#include "base/util/env.h"
#include "base/util/logging.h"
#include "base/util/uuid.h"
#include "tile/hal/opencl/cl_opt.h"
#include "tile/hal/opencl/emitocl.h"
#include "tile/hal/opencl/library.h"
#include "tile/lang/semprinter.h"

namespace fs = boost::filesystem;

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {
namespace {

// Represents a build-in-flight
class Build {
 public:
  static boost::future<std::unique_ptr<hal::Library>> Start(context::Activity activity,
                                                            const std::shared_ptr<DeviceState>& device_state,
                                                            CLObj<cl_program> program,
                                                            const std::vector<lang::KernelInfo>& kernel_info,
                                                            proto::BuildInfo binfo,
                                                            std::vector<context::proto::ActivityID> kernel_ids);

  Build(context::Activity activity, const std::shared_ptr<DeviceState>& device_state, CLObj<cl_program> program,
        const std::vector<lang::KernelInfo>& kernel_info, proto::BuildInfo binfo,
        std::vector<context::proto::ActivityID> kernel_ids);

 private:
  static void OnBuildComplete(cl_program program, void* handle) noexcept;

  void OnError();

  context::Activity activity_;
  std::shared_ptr<DeviceState> device_state_;
  std::unique_ptr<Library> library_;
  boost::promise<std::unique_ptr<hal::Library>> prom_;
  proto::BuildInfo binfo_;

  static PendingCallbackMap<Build> pending_;
};

PendingCallbackMap<Build> Build::pending_;

boost::future<std::unique_ptr<hal::Library>> Build::Start(context::Activity activity,
                                                          const std::shared_ptr<DeviceState>& device_state,
                                                          CLObj<cl_program> program,
                                                          const std::vector<lang::KernelInfo>& kernel_info,
                                                          proto::BuildInfo binfo,
                                                          std::vector<context::proto::ActivityID> kernel_ids) {
  auto build = compat::make_unique<Build>(std::move(activity), device_state, std::move(program), kernel_info,
                                          std::move(binfo), std::move(kernel_ids));
  auto result = build->prom_.get_future();
  cl_device_id device_id = device_state->did();
  cl_program prog = build->library_->program().get();
  auto handle = Build::pending_.Acquire(std::move(build));
  Err err = clBuildProgram(prog, 1, &device_id, "-cl-fast-relaxed-math -cl-mad-enable -cl-unsafe-math-optimizations",
                           &OnBuildComplete, handle);
  if (err) {
    LOG(WARNING) << "Failed to build program: " << err;
    OnBuildComplete(prog, handle);
  }

  return result;
}

Build::Build(context::Activity activity, const std::shared_ptr<DeviceState>& device_state, CLObj<cl_program> program,
             const std::vector<lang::KernelInfo>& kernel_info, proto::BuildInfo binfo,
             std::vector<context::proto::ActivityID> kernel_ids)
    : activity_{std::move(activity)},
      device_state_{device_state},
      library_{compat::make_unique<Library>(device_state, std::move(program), kernel_info, std::move(kernel_ids))},
      binfo_{std::move(binfo)} {}

void Build::OnBuildComplete(cl_program program, void* handle) noexcept {
  auto build = Build::pending_.Release(handle);
  if (!build) {
    // no-op, this handle has already been processed.
    return;
  }

  try {
    cl_build_status status;
    Err::Check(clGetProgramBuildInfo(build->library_->program().get(), build->device_state_->did(),
                                     CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, nullptr),
               "Unable to construct program build status");
    if (status == CL_BUILD_SUCCESS) {
      build->prom_.set_value(std::move(build->library_));
    } else {
      LOG(WARNING) << "Failed to build program";
      build->binfo_.set_cl_build_status(status);
      build->OnError();
    }
    build->activity_.AddMetadata(build->binfo_);
  } catch (...) {
    build->prom_.set_exception(boost::current_exception());
  }
}

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

void Build::OnError() {
  size_t len = 0;
  Err bi_err =
      clGetProgramBuildInfo(library_->program().get(), device_state_->did(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
  if (bi_err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to retrieve build log size: " << bi_err;
  } else {
    std::string buffer(len, '\0');
    bi_err = clGetProgramBuildInfo(library_->program().get(), device_state_->did(), CL_PROGRAM_BUILD_LOG, len,
                                   const_cast<char*>(buffer.c_str()), nullptr);
    if (bi_err) {
      LOG(ERROR) << "Failed to retrieve build log: " << bi_err;
    } else {
      LOG(WARNING) << "Failed build log: " << buffer;
      LOG(WARNING) << "Code was: \n" << WithLineNumbers(binfo_.src());
      binfo_.set_log(buffer);
    }
  }
  throw std::runtime_error{"Unable to compile Tile program"};
}

std::string ReadFile(const fs::path& path) {
  fs::ifstream ifs;
  ifs.open(path);
  auto it = std::istreambuf_iterator<char>(ifs);
  auto it_end = std::istreambuf_iterator<char>();
  std::string contents(it, it_end);
  if (ifs.bad()) {
    throw std::runtime_error("Unable to fully read file: " + path.string());
  }
  return contents;
}

void WriteFile(const fs::path& path, const std::string& contents) {
  fs::ofstream ofs(path);
  ofs << contents;
}

}  // namespace

Compiler::Compiler(const std::shared_ptr<DeviceState>& device_state) : device_state_{device_state} {}

boost::future<std::unique_ptr<hal::Library>> Compiler::Build(const context::Context& ctx,
                                                             const std::vector<lang::KernelInfo>& kernel_info,
                                                             const hal::proto::HardwareSettings& settings) {
  std::vector<context::proto::ActivityID> kernel_ids;
  std::ostringstream code;

  if (!kernel_info.size()) {
    return boost::make_ready_future(std::unique_ptr<hal::Library>{
        compat::make_unique<Library>(device_state_, nullptr, kernel_info, std::vector<context::proto::ActivityID>{})});
  }

  context::Activity activity{ctx, "tile::hal::opencl::Build"};

  bool cl_khr_fp16 = device_state_->HasDeviceExtension("cl_khr_fp16");
  if (cl_khr_fp16) {
    code << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
  }

  bool cl_khr_fp64 = device_state_->HasDeviceExtension("cl_khr_fp64");
  if (cl_khr_fp64) {
    code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
  }

  auto env_cache = env::Get("PLAIDML_OPENCL_CACHE");
  fs::path cache_dir;
  if (env_cache.length()) {
    VLOG(1) << "Using OpenCL cache directory: " << env_cache;
    cache_dir = env_cache;
  }
  std::set<std::string> knames;

  for (const auto& ki : kernel_info) {
    context::Activity kbuild{activity.ctx(), "tile::hal::opencl::BuildKernel"};

    proto::KernelInfo kinfo;
    kinfo.set_kname(ki.kname);

    if (ki.ktype == lang::KernelType::kZero) {
      kinfo.set_src("// Builtin zero kernel");
    } else if (!knames.count(ki.kfunc->name)) {
      knames.insert(ki.kfunc->name);
      OptimizeKernel(ki, cl_khr_fp16, settings);

      Emit ocl{cl_khr_fp16, cl_khr_fp64};
      ocl.Visit(*ki.kfunc);
      std::string src = ki.comments + ocl.str();

      if (is_directory(cache_dir)) {
        fs::path src_path = (cache_dir / ki.kname).replace_extension("cl");
        if (fs::is_regular_file(src_path)) {
          VLOG(1) << "Reading OpenCL code from cache: " << src_path;
          src = ReadFile(src_path);
        } else {
          VLOG(1) << "Writing OpenCL code to cache: " << src_path;
          WriteFile(src_path, src);
        }
      } else {
        if (VLOG_IS_ON(4)) {
          sem::Print emit_debug(*ki.kfunc);
          VLOG(4) << "Generic debug kernel:";
          VLOG(4) << ki.comments;
          VLOG(4) << emit_debug.str();
        }
      }

      code << src;
      code << "\n\n";

      kinfo.set_src(src);
    } else {
      kinfo.set_src("// Duplicate");
    }

    *(kinfo.mutable_kinfo()) = ki.info;
    kbuild.AddMetadata(kinfo);

    kernel_ids.emplace_back(kbuild.ctx().activity_id());
  }

  proto::BuildInfo binfo;
  *binfo.mutable_device_id() = device_state_->id();
  binfo.set_src(code.str());
  const char* src = binfo.src().c_str();
  Err err;

  VLOG(4) << "Compiling OpenCL:\n" << WithLineNumbers(binfo.src());
  CLObj<cl_program> program = clCreateProgramWithSource(device_state_->cl_ctx().get(), 1, &src, nullptr, err.ptr());
  if (!program) {
    throw std::runtime_error(std::string("creating an OpenCL program object: ") + err.str());
  }

  return Build::Start(std::move(activity), device_state_, std::move(program), kernel_info, std::move(binfo),
                      std::move(kernel_ids));
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
