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
#include "base/util/file.h"
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
  auto build = std::make_unique<Build>(std::move(activity), device_state, std::move(program), kernel_info,
                                       std::move(binfo), std::move(kernel_ids));
  auto result = build->prom_.get_future();
  cl_device_id device_id = device_state->did();
  cl_program prog = build->library_->program().get();
  auto handle = Build::pending_.Acquire(std::move(build));
  Err err = ocl::BuildProgram(prog, 1, &device_id, "-cl-fast-relaxed-math -cl-mad-enable -cl-unsafe-math-optimizations",
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
      library_{std::make_unique<Library>(device_state, std::move(program), kernel_info, std::move(kernel_ids))},
      binfo_{std::move(binfo)} {}

void Build::OnBuildComplete(cl_program program, void* handle) noexcept {
  auto build = Build::pending_.Release(handle);
  if (!build) {
    // no-op, this handle has already been processed.
    return;
  }

  try {
    cl_build_status status;
    Err::Check(ocl::GetProgramBuildInfo(build->library_->program().get(), build->device_state_->did(),
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
      ocl::GetProgramBuildInfo(library_->program().get(), device_state_->did(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
  if (bi_err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to retrieve build log size: " << bi_err;
  } else {
    std::string buffer(len, '\0');
    bi_err = ocl::GetProgramBuildInfo(library_->program().get(), device_state_->did(), CL_PROGRAM_BUILD_LOG, len,
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

}  // namespace

Compiler::Compiler(const std::shared_ptr<DeviceState>& device_state) : device_state_{device_state} {}

std::string k_subgroup_microkernels =  // NOLINT
    R"***(

#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define ITERATION( _index, _src1_lda ) \
        {   \
            const float4 a0 = intel_sub_group_shuffle( arow0, _index ); \
            const float4 a1 = intel_sub_group_shuffle( arow1, _index ); \
            const float4 a2 = intel_sub_group_shuffle( arow2, _index ); \
            const float4 a3 = intel_sub_group_shuffle( arow3, _index ); \
            const float4 a4 = intel_sub_group_shuffle( arow4, _index ); \
            const float4 a5 = intel_sub_group_shuffle( arow5, _index ); \
            const float4 a6 = intel_sub_group_shuffle( arow6, _index ); \
            const float4 a7 = intel_sub_group_shuffle( arow7, _index ); \
            const float4 brow00 = src1_read0[ 0 ];   src1_read0 += _src1_lda;    \
            const float4 brow01 = src1_read0[ 0 ];   src1_read0 += _src1_lda;    \
            const float4 brow02 = src1_read0[ 0 ];   src1_read0 += _src1_lda;    \
            const float4 brow03 = src1_read0[ 0 ];   src1_read0 += _src1_lda;    \
            dot00 = mad(brow00, (float4) a0.x, dot00);  \
            dot00 = mad(brow01, (float4) a0.y, dot00);  \
            dot00 = mad(brow02, (float4) a0.z, dot00);  \
            dot00 = mad(brow03, (float4) a0.w, dot00);  \
            dot01 = mad(brow00, (float4) a1.x, dot01);  \
            dot01 = mad(brow01, (float4) a1.y, dot01);  \
            dot01 = mad(brow02, (float4) a1.z, dot01);  \
            dot01 = mad(brow03, (float4) a1.w, dot01);  \
            dot02 = mad(brow00, (float4) a2.x, dot02);  \
            dot02 = mad(brow01, (float4) a2.y, dot02);  \
            dot02 = mad(brow02, (float4) a2.z, dot02);  \
            dot02 = mad(brow03, (float4) a2.w, dot02);  \
            dot03 = mad(brow00, (float4) a3.x, dot03);  \
            dot03 = mad(brow01, (float4) a3.y, dot03);  \
            dot03 = mad(brow02, (float4) a3.z, dot03);  \
            dot03 = mad(brow03, (float4) a3.w, dot03);  \
            dot04 = mad(brow00, (float4) a4.x, dot04);  \
            dot04 = mad(brow01, (float4) a4.y, dot04);  \
            dot04 = mad(brow02, (float4) a4.z, dot04);  \
            dot04 = mad(brow03, (float4) a4.w, dot04);  \
            dot05 = mad(brow00, (float4) a5.x, dot05);  \
            dot05 = mad(brow01, (float4) a5.y, dot05);  \
            dot05 = mad(brow02, (float4) a5.z, dot05);  \
            dot05 = mad(brow03, (float4) a5.w, dot05);  \
            dot06 = mad(brow00, (float4) a6.x, dot06);  \
            dot06 = mad(brow01, (float4) a6.y, dot06);  \
            dot06 = mad(brow02, (float4) a6.z, dot06);  \
            dot06 = mad(brow03, (float4) a6.w, dot06);  \
            dot07 = mad(brow00, (float4) a7.x, dot07);  \
            dot07 = mad(brow01, (float4) a7.y, dot07);  \
            dot07 = mad(brow02, (float4) a7.z, dot07);  \
            dot07 = mad(brow03, (float4) a7.w, dot07);  \
        }

#define DO_MAC_X(dst, dst_off, dst_lda, src0, src0_off, src0_lda, src1, src1_off, src1_lda)
    // NO WORK 

#define MAC_INIT() \
    float4 dot00 = (float4)(0.f); \
    float4 dot01 = (float4)(0.f); \
    float4 dot02 = (float4)(0.f); \
    float4 dot03 = (float4)(0.f); \
    float4 dot04 = (float4)(0.f); \
    float4 dot05 = (float4)(0.f); \
    float4 dot06 = (float4)(0.f); \
    float4 dot07 = (float4)(0.f);

#define MAC_FINISH(dst, dst_off, dst_lda) \
    __global float4 *dst_write0 = ((__global float4 *) (dst + dst_off)) + get_local_id(0) % 8; \
    dst_write0[ 0 ] = dot00;  dst_write0 += dst_lda; \
    dst_write0[ 0 ] = dot01;  dst_write0 += dst_lda; \
    dst_write0[ 0 ] = dot02;  dst_write0 += dst_lda; \
    dst_write0[ 0 ] = dot03;  dst_write0 += dst_lda; \
    dst_write0[ 0 ] = dot04;  dst_write0 += dst_lda; \
    dst_write0[ 0 ] = dot05;  dst_write0 += dst_lda; \
    dst_write0[ 0 ] = dot06;  dst_write0 += dst_lda; \
    dst_write0[ 0 ] = dot07;  dst_write0 += dst_lda; 

#define MAC_INNER(src0, src0_off, src0_lda, src1, src1_off, src1_lda) \
    const __global float4 *src0_read = ((const __global float4 *) (src0 + src0_off)) + get_local_id(0) % 8; \
    const __global float4 *src1_read0 = ((const __global float4 *) (src1 + src1_off)) + get_local_id(0) % 8; \
    int w = 0; \
    do \
    { \
        const float4 arow0 = src0_read[ 0 * src0_lda]; \
        const float4 arow1 = src0_read[ 1 * src0_lda ]; \
        const float4 arow2 = src0_read[ 2 * src0_lda ]; \
        const float4 arow3 = src0_read[ 3 * src0_lda ]; \
        const float4 arow4 = src0_read[ 4 * src0_lda ]; \
        const float4 arow5 = src0_read[ 5 * src0_lda ]; \
        const float4 arow6 = src0_read[ 6 * src0_lda ]; \
        const float4 arow7 = src0_read[ 7 * src0_lda ]; \
        ITERATION( 0, src1_lda ); \
        ITERATION( 1, src1_lda ); \
        ITERATION( 2, src1_lda ); \
        ITERATION( 3, src1_lda ); \
        ITERATION( 4, src1_lda ); \
        ITERATION( 5, src1_lda ); \
        ITERATION( 6, src1_lda ); \
        ITERATION( 7, src1_lda ); \
        src0_read += 8; \
        w += 8; \
    } \
    while( w < src0_lda );

)***";                                 // NOLINT

boost::future<std::unique_ptr<hal::Library>> Compiler::Build(const context::Context& ctx,
                                                             const std::vector<lang::KernelInfo>& kernel_info,
                                                             const hal::proto::HardwareSettings& settings) {
  std::vector<context::proto::ActivityID> kernel_ids;
  std::ostringstream code;

  if (!kernel_info.size()) {
    return boost::make_ready_future(std::unique_ptr<hal::Library>{
        std::make_unique<Library>(device_state_, nullptr, kernel_info, std::vector<context::proto::ActivityID>{})});
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

  bool cl_intel_subgroups = device_state_->HasDeviceExtension("cl_intel_subgroups");
  if (cl_intel_subgroups) {
    code << k_subgroup_microkernels;
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
  CLObj<cl_program> program = ocl::CreateProgramWithSource(device_state_->cl_ctx().get(), 1, &src, nullptr, err.ptr());
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
