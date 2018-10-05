// Copyright 2018, Intel Corporation.

#include "cuda/include/nvrtc.h"
#include "tile/hal/cuda/emit.h"
#include "tile/hal/cuda/error.h"
#include "tile/hal/cuda/hal.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cuda {

namespace nvrtc {

class Error {
 public:
  Error(nvrtcResult code)  // NOLINT
      : code_{code} {}

  static void Check(Error err, const std::string& msg) {
    if (err) {
      throw std::runtime_error(msg + ": " + err.str());
    }
  }

  operator bool() const { return code_ != NVRTC_SUCCESS; }

  nvrtcResult code() const { return code_; }

  std::string str() const { return nvrtcGetErrorString(code_); }

 private:
  nvrtcResult code_;
};

}  // namespace nvrtc

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

Compiler::Compiler(Device* device) : device_(device) {}

boost::future<std::unique_ptr<hal::Library>> Compiler::Build(const context::Context& ctx,
                                                             const std::vector<lang::KernelInfo>& kernels,
                                                             const hal::proto::HardwareSettings& settings) {
  if (!kernels.size()) {
    return boost::make_ready_future(
        std::unique_ptr<hal::Library>{compat::make_unique<Library>(std::vector<std::shared_ptr<Kernel>>{})});
  }

  auto src = EmitCudaC(kernels);
  VLOG(3) << "Compiling CUDA C:\n" << WithLineNumbers(src);

  nvrtcProgram program;
  nvrtc::Error nvrtc_err = nvrtcCreateProgram(&program,     // prog
                                              src.c_str(),  // src
                                              nullptr,      // name
                                              0,            // numHeaders
                                              nullptr,      // headers
                                              nullptr);     // includeNames
  nvrtc::Error::Check(nvrtc_err, "nvrtcCreateProgram() failed");

  nvrtc::Error compile_err = nvrtcCompileProgram(program, 0, nullptr);

  size_t log_size;
  nvrtc_err = nvrtcGetProgramLogSize(program, &log_size);
  nvrtc::Error::Check(nvrtc_err, "nvrtcGetProgramLogSize() failed");

  std::string log;
  log.resize(log_size);
  nvrtc_err = nvrtcGetProgramLog(program, &log[0]);
  nvrtc::Error::Check(nvrtc_err, "nvrtcGetProgramLog() failed");

  if (compile_err) {
    LOG(ERROR) << "NVRTC build log: " << log;
    nvrtc::Error::Check(compile_err, "nvrtcCompileProgram() failed");
  } else {
    VLOG(3) << "NVRTC build log: " << log;
  }

  size_t ptx_size;
  nvrtc_err = nvrtcGetPTXSize(program, &ptx_size);
  nvrtc::Error::Check(nvrtc_err, "nvrtcGetPTXSize() failed");

  VLOG(2) << "PTX: " << ptx_size << " bytes";

  std::vector<char> ptx;
  ptx.resize(ptx_size);
  nvrtc_err = nvrtcGetPTX(program, ptx.data());
  nvrtc::Error::Check(nvrtc_err, "nvrtcGetPTX() failed");

  nvrtc_err = nvrtcDestroyProgram(&program);
  nvrtc::Error::Check(nvrtc_err, "nvrtcDestroyProgram() failed");

  device_->SetCurrentContext();

  CUmodule module;
  Error cuda_err = cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0);
  Error::Check(cuda_err, "cuModuleLoadDataEx() failed");

  std::vector<std::shared_ptr<Kernel>> result;
  for (const auto& ki : kernels) {
    if (ki.ktype == lang::KernelType::kZero) {
      result.emplace_back(std::make_shared<ZeroKernel>(device_, ki));
    } else {
      CUfunction function;
      cuda_err = cuModuleGetFunction(&function, module, ki.kname.c_str());
      Error::Check(cuda_err, "cuModuleGetFunction() failed");
      result.emplace_back(std::make_shared<CodeKernel>(device_, ki, function));
    }
  }
  std::unique_ptr<hal::Library> lib(new Library(std::move(result)));
  return boost::make_ready_future(std::move(lib));
}

}  // namespace cuda
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
