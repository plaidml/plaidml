// Copyright 2017, Vertex.AI.

#include "tile/hal/opencl/compiler.h"

#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/filesystem.hpp>

#include "base/util/compat.h"
#include "base/util/logging.h"
#include "base/util/uuid.h"
#include "tile/hal/opencl/emitocl.h"
#include "tile/hal/opencl/library.h"

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
                                                            std::vector<boost::uuids::uuid> kernel_uuids);

  Build(context::Activity activity, const std::shared_ptr<DeviceState>& device_state, CLObj<cl_program> program,
        const std::vector<lang::KernelInfo>& kernel_info, proto::BuildInfo binfo,
        std::vector<boost::uuids::uuid> kernel_uuids);

 private:
  static void OnBuildComplete(cl_program program, void* raw_build) noexcept;

  void OnError() noexcept;

  context::Activity activity_;
  std::shared_ptr<DeviceState> device_state_;
  std::unique_ptr<Library> library_;
  boost::promise<std::unique_ptr<hal::Library>> prom_;
  proto::BuildInfo binfo_;
};

boost::future<std::unique_ptr<hal::Library>> Build::Start(context::Activity activity,
                                                          const std::shared_ptr<DeviceState>& device_state,
                                                          CLObj<cl_program> program,
                                                          const std::vector<lang::KernelInfo>& kernel_info,
                                                          proto::BuildInfo binfo,
                                                          std::vector<boost::uuids::uuid> kernel_uuids) {
  auto build = compat::make_unique<Build>(std::move(activity), device_state, std::move(program), kernel_info,
                                          std::move(binfo), std::move(kernel_uuids));
  auto result = build->prom_.get_future();

  cl_device_id device_id = device_state->did();
  cl_program prog = build->library_->program().get();
  Err err = clBuildProgram(prog, 1, &device_id, "-w -cl-fast-relaxed-math -cl-mad-enable -cl-unsafe-math-optimizations",
                           &OnBuildComplete, build.release());
  if (err) {
    LOG(WARNING) << "Failed to build program: " << err;
    // N.B. OnBuildComplete will be called by OpenCL at some point, even for failed builds.
  }

  return result;
}

Build::Build(context::Activity activity, const std::shared_ptr<DeviceState>& device_state, CLObj<cl_program> program,
             const std::vector<lang::KernelInfo>& kernel_info, proto::BuildInfo binfo,
             std::vector<boost::uuids::uuid> kernel_uuids)
    : activity_{std::move(activity)},
      device_state_{device_state},
      library_{compat::make_unique<Library>(device_state, std::move(program), kernel_info, std::move(kernel_uuids))},
      binfo_{std::move(binfo)} {}

void Build::OnBuildComplete(cl_program program, void* raw_build) noexcept {
  std::unique_ptr<Build> build{static_cast<Build*>(raw_build)};

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
}

void Build::OnError() noexcept {
  try {
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
        std::stringstream ss_in(binfo_.src());
        std::stringstream ss_out;
        size_t line_num = 1;
        std::string line;
        while (std::getline(ss_in, line, '\n')) {
          ss_out << std::setw(5) << line_num++ << ": " << line << "\n";
        }
        LOG(WARNING) << "Code was: \n" << ss_out.str();
        binfo_.set_log(buffer);
      }
    }
    throw std::runtime_error{"Unable to compile Tile program"};
  } catch (...) {
    prom_.set_exception(boost::current_exception());
  }
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
  std::vector<boost::uuids::uuid> kernel_uuids;
  std::ostringstream code;

  context::Activity activity{ctx, "tile::hal::opencl::Build"};

  bool cl_khr_fp16 = device_state_->HasDeviceExtension("cl_khr_fp16");
  if (cl_khr_fp16) {
    code << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
  }

  bool cl_khr_fp64 = device_state_->HasDeviceExtension("cl_khr_fp64");
  if (cl_khr_fp64) {
    code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
  }

  auto env_cache = std::getenv("PLAIDML_OPENCL_CACHE");
  fs::path cache_dir;
  if (env_cache) {
    VLOG(1) << "Using OpenCL cache directory: " << env_cache;
    cache_dir = env_cache;
  }

  for (const auto& ki : kernel_info) {
    context::Activity kbuild{activity.ctx(), "tile::hal::opencl::BuildKernel"};

    if (ki.ktype == lang::KernelType::kZero) {
      continue;
    }

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
        lang::EmitDebug emit_debug;
        emit_debug.Visit(*ki.kfunc);
        VLOG(4) << "Generic debug kernel:";
        VLOG(4) << ki.comments;
        VLOG(4) << emit_debug.str();
      }
    }

    code << src;
    code << "\n\n";

    proto::KernelInfo kinfo;
    kinfo.set_kname(ki.kname);
    kinfo.set_src(src);
    *(kinfo.mutable_kinfo()) = ki.info;
    kbuild.AddMetadata(kinfo);

    kernel_uuids.emplace_back(kbuild.ctx().activity_uuid());
  }

  proto::BuildInfo binfo;
  binfo.set_device_uuid(ToByteString(device_state_->uuid()));
  binfo.set_src(code.str());
  const char* src = binfo.src().c_str();
  Err err;

  VLOG(4) << "Compiling OpenCL:\n" << binfo.src();
  CLObj<cl_program> program = clCreateProgramWithSource(device_state_->cl_ctx().get(), 1, &src, nullptr, err.ptr());
  if (!program) {
    throw std::runtime_error(std::string("creating an OpenCL program object: ") + err.str());
  }

  return Build::Start(std::move(activity), device_state_, std::move(program), kernel_info, std::move(binfo),
                      std::move(kernel_uuids));
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
