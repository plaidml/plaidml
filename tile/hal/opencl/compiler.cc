// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/opencl/compiler.h"

#include <exception>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

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

int Compiler::i = 0;

namespace {

struct BuildState;

// Represents a build-in-flight
class Build {
 public:
  Build(context::Activity activity, const std::shared_ptr<DeviceState>& device_state,
        const std::map<std::string, CLObj<cl_program>>& program, const std::vector<lang::KernelInfo>& kernel_info,
        const std::map<std::string, proto::BuildInfo>& binfo, std::vector<context::proto::ActivityID> kernel_ids);

  boost::future<std::unique_ptr<hal::Library>> Start();
  std::unique_ptr<Library>& library() { return library_; }
  std::shared_ptr<DeviceState>& device_state() { return device_state_; }

  static void CompileKernel(std::shared_ptr<BuildState> build_state);
  static void OnBuildComplete(cl_program program, void* handle) noexcept;

 private:
  void OnError(const std::string& current);
  context::Activity activity_;
  std::shared_ptr<DeviceState> device_state_;
  std::unique_ptr<Library> library_;
  boost::promise<std::unique_ptr<hal::Library>> prom_;
  std::map<std::string, proto::BuildInfo> binfo_;
};

struct BuildState {
  BuildState(Build* b, const std::string& c) : build(b), current(c) {}
  Build* build;
  std::string current;
};

// Compile single block/kernel

void Build::CompileKernel(std::shared_ptr<BuildState> build_state) {
  auto prog_it = build_state->build->library()->program().find(build_state->current);
  cl_device_id device_id = build_state->build->device_state()->did();
  clock_t build_start = clock();
  Err err = ocl::BuildProgram(prog_it->second.get(), 1, &device_id,
                              "-cl-fast-relaxed-math -cl-mad-enable -cl-unsafe-math-optimizations", &OnBuildComplete,
                              reinterpret_cast<void*>(build_state.get()));
  clock_t build_end = clock();
  if (env::Get("PLAIDML_BUILD_TIMES") == "1") {
    double elapsed_secs = static_cast<double>(build_end - build_start) / CLOCKS_PER_SEC;
    std::cout << "Built " << prog_it->first << " in " << elapsed_secs << " seconds.\n";
  }
  if (err) {
    LOG(WARNING) << "Failed to build program " << prog_it->first << ": " << err;
    OnBuildComplete(prog_it->second.get(), reinterpret_cast<void*>(build_state.get()));
  }
}

boost::future<std::unique_ptr<hal::Library>> Build::Start() {
  auto result = prom_.get_future();
  boost::thread_group threads;
  boost::asio::io_service io_service;

  // Allocate tasks
  auto& program_map = library_->program();
  clock_t build_start = clock();
  for (auto& prog_it : program_map) {
    auto bs = std::make_shared<BuildState>(this, prog_it.first);
    io_service.post(boost::bind(&(Build::CompileKernel), bs));
  }
  // Create thread pool and threads
  unsigned int n_threads = (env::Get("OPENCL_BUILD_THREADS") == "")
                               ? boost::thread::hardware_concurrency()
                               : std::atoi(env::Get("OPENCL_BUILD_THREADS").c_str());
  for (size_t i = 0; i < n_threads; ++i) {
    threads.create_thread(boost::bind(&boost::asio::io_service::run, &io_service));
  }

  threads.join_all();
  io_service.stop();
  clock_t build_end = clock();
  if (env::Get("PLAIDML_BUILD_TIMES") == "1") {
    double elapsed_secs = static_cast<double>(build_end - build_start) / CLOCKS_PER_SEC;
    std::cout << "Total compilation time: " << elapsed_secs << " seconds\n";
  }
  prom_.set_value(std::move(library_));
  return result;
}

Build::Build(context::Activity activity, const std::shared_ptr<DeviceState>& device_state,
             const std::map<std::string, CLObj<cl_program>>& program, const std::vector<lang::KernelInfo>& kernel_info,
             const std::map<std::string, proto::BuildInfo>& binfo, std::vector<context::proto::ActivityID> kernel_ids)
    : activity_{std::move(activity)},
      device_state_{device_state},
      library_{std::make_unique<Library>(device_state, std::move(program), kernel_info, std::move(kernel_ids))},
      binfo_{std::move(binfo)} {}

void Build::OnBuildComplete(cl_program program, void* handle) noexcept {
  BuildState* build_state = static_cast<BuildState*>(handle);
  if (!build_state) {
    // no-op, this handle has already been processed.
    return;
  }

  Build* build = build_state->build;
  try {
    cl_build_status status;
    Err::Check(ocl::GetProgramBuildInfo(program, build->device_state()->did(), CL_PROGRAM_BUILD_STATUS, sizeof(status),
                                        &status, nullptr),
               "Unable to construct program build status");
    if (status != CL_BUILD_SUCCESS) {
      LOG(WARNING) << "Failed to build program " << build_state->current;
      build->binfo_[build_state->current].set_cl_build_status(status);
      build->OnError(build_state->current);
    }
    build->activity_.AddMetadata(build->binfo_[build_state->current]);
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

void Build::OnError(const std::string& current) {
  size_t len = 0;
  auto prog_it = library_->program().find(current);

  Err bi_err =
      ocl::GetProgramBuildInfo(prog_it->second.get(), device_state_->did(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
  if (bi_err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to retrieve build log size for" << prog_it->first << ": " << bi_err;
  } else {
    std::string buffer(len, '\0');
    bi_err = ocl::GetProgramBuildInfo(prog_it->second.get(), device_state_->did(), CL_PROGRAM_BUILD_LOG, len,
                                      const_cast<char*>(buffer.c_str()), nullptr);
    if (bi_err) {
      LOG(ERROR) << "Failed to retrieve build log for" << prog_it->first << ": " << bi_err;
    } else {
      auto& bi = binfo_[prog_it->first];
      LOG(WARNING) << "Failed build log: " << buffer;
      LOG(WARNING) << "Code was: \n" << WithLineNumbers(bi.src());
      bi.set_log(buffer);
    }
  }

  throw std::runtime_error{"Unable to compile Tile program"};
}

}  // namespace

Compiler::Compiler(const std::shared_ptr<DeviceState>& device_state) : device_state_{device_state} {}

std::string k_subgroup_microkernels =  // NOLINT
    R"***(

#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define vector_load(x) as_float(intel_sub_group_block_read((const global int*) (&(x))))
#define vector_store(x, v) intel_sub_group_block_write((const global int*) (&(x)), as_uint(v))

)***";                                 // NOLINT

boost::future<std::unique_ptr<hal::Library>> Compiler::Build(const context::Context& ctx,
                                                             const std::vector<lang::KernelInfo>& kernel_info,
                                                             const hal::proto::HardwareSettings& settings) {
  std::vector<context::proto::ActivityID> kernel_ids;
  std::ostringstream header;

  if (!kernel_info.size()) {
    return boost::make_ready_future(std::unique_ptr<hal::Library>{
        std::make_unique<Library>(device_state_, std::map<std::string, CLObj<cl_program>>{}, kernel_info,
                                  std::vector<context::proto::ActivityID>{})});
  }

  context::Activity activity{ctx, "tile::hal::opencl::Build"};

  if (env::Get("PLAIDML_SPIRV_BACKEND") == "1") {
    std::set<std::string> knames;
    std::map<std::string, CLObj<cl_program>> program_map;
    std::map<std::string, proto::BuildInfo> binfo_map;

    for (const auto& ki : kernel_info) {
      context::Activity kbuild{activity.ctx(), "tile::hal::opencl::BuildKernel"};
      proto::KernelInfo kinfo;
      kinfo.set_kname(ki.kname);

      if (ki.ktype == lang::KernelType::kZero) {
        kinfo.set_src("// Builtin zero kernel");
      } else if (!knames.count(ki.kfunc->name)) {
        knames.insert(ki.kfunc->name);

        auto kname = ki.kname + "_" + std::to_string(i);
        i++;

        fs::path spv_binary_path = env::Get("PLAIDML_SPIRV_CACHE");
        auto spv_binary_suffix = env::Get("PLAIDML_SPIRV_BINARY_SUFFIX");
        auto spv_binary_file = spv_binary_path / (kname + spv_binary_suffix);

        VLOG(1) << "Reading SPIRV_BINARY to OpenCL backend: " << spv_binary_file << std::endl;

        auto spv_binary_str = ReadFile(spv_binary_file, true);
        const void* spv_binary = spv_binary_str.c_str();

        Err err;
        VLOG(1) << "CreateProgramWithIL " << spv_binary_file;
        CLObj<cl_program> program =
            ocl::CreateProgramWithIL(device_state_->cl_ctx().get(), spv_binary, spv_binary_str.size(), err.ptr());

        if (!program) {
          throw std::runtime_error(std::string("Creating an OpenCL program object from  ") + spv_binary_file.c_str() + ": " +
                                   err.str());
        }

        proto::BuildInfo binfo;
        *binfo.mutable_device_id() = device_state_->id();

        program_map.emplace(ki.kname, std::move(program));
        binfo_map.emplace(ki.kname, std::move(binfo));
      } else {
        kinfo.set_src("// Duplicate");
      }

      *(kinfo.mutable_kinfo()) = ki.info;
      kbuild.AddMetadata(kinfo);

      kernel_ids.emplace_back(kbuild.ctx().activity_id());
    }

    opencl::Build build(std::move(activity), device_state_, std::move(program_map), kernel_info, std::move(binfo_map),
                        std::move(kernel_ids));
    return build.Start();
  } else {
    bool cl_khr_fp16 = device_state_->HasDeviceExtension("cl_khr_fp16");
    if (cl_khr_fp16) {
      header << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    }

    bool cl_khr_fp64 = device_state_->HasDeviceExtension("cl_khr_fp64");
    if (cl_khr_fp64) {
      header << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    }

    bool cl_intel_subgroups = device_state_->HasDeviceExtension("cl_intel_subgroups");
    if (cl_intel_subgroups) {
      header << k_subgroup_microkernels;
    }

    auto env_cache = env::Get("PLAIDML_OPENCL_CACHE");
    fs::path cache_dir;
    if (env_cache.length()) {
      VLOG(1) << "Using OpenCL cache directory: " << env_cache;
      cache_dir = env_cache;
    }
    std::set<std::string> knames;

    std::map<std::string, CLObj<cl_program>> program_map;
    std::map<std::string, proto::BuildInfo> binfo_map;

    for (const auto& ki : kernel_info) {
      std::ostringstream code;
      code << header.str();
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

        std::stringstream src;
        src << "// gid: " << ki.gwork[0] << " " << ki.gwork[1] << " " << ki.gwork[2] << "\n";
        src << "// lid: " << ki.lwork[0] << " " << ki.lwork[1] << " " << ki.lwork[2] << "\n";
        src << ki.comments << ocl.str();

        auto kname = ki.kname + "_" + std::to_string(i);
        i++;
        if (is_directory(cache_dir)) {
          fs::path src_path = (cache_dir / kname).replace_extension("cl");
          if (fs::is_regular_file(src_path)) {
            VLOG(1) << "Reading OpenCL code from cache: " << src_path;
            // src.clear();
            // src << ReadFile(src_path);
          } else {
            VLOG(1) << "Writing OpenCL code to cache: " << src_path;
            WriteFile(src_path, src.str());
          }
        } else {
          if (VLOG_IS_ON(4)) {
            sem::Print emit_debug(*ki.kfunc);
            VLOG(4) << "Generic debug kernel:";
            VLOG(4) << ki.comments;
            VLOG(4) << emit_debug.str();
          }
        }

        code << src.str();
        kinfo.set_src(src.str());
        proto::BuildInfo binfo;
        *binfo.mutable_device_id() = device_state_->id();
        binfo.set_src(code.str());
        const char* buf = binfo.src().c_str();
        // Output the OpenCL source code to file
        std::string out_dir = env::Get("PLAIDML_OPENCL_OUTPUT");
        if (out_dir.size() > 0) {
          fs::path out_path = out_dir;
          fs::path src_path = (out_path / kname).replace_extension("cl");
          WriteFile(src_path, code.str());
        }
        Err err;

        CLObj<cl_program> program =
            ocl::CreateProgramWithSource(device_state_->cl_ctx().get(), 1, &buf, nullptr, err.ptr());

        if (!program) {
          throw std::runtime_error(std::string("Creating an OpenCL program object for ") + ki.kname + ": " + err.str());
        }
        program_map.emplace(ki.kname, std::move(program));
        binfo_map.emplace(ki.kname, std::move(binfo));
      } else {
        kinfo.set_src("// Duplicate");
      }

      *(kinfo.mutable_kinfo()) = ki.info;
      kbuild.AddMetadata(kinfo);

      kernel_ids.emplace_back(kbuild.ctx().activity_id());
    }

    opencl::Build build(std::move(activity), device_state_, std::move(program_map), kernel_info, std::move(binfo_map),
                        std::move(kernel_ids));
    return build.Start();
  }
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
