// Copyright 2018, Intel Corporation.

#pragma once

#include <string>

#include "tile/hal/cuda/hal.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cuda {

class Error {
 public:
  Error(CUresult code)  // NOLINT
      : code_{code} {}

  static void Check(Error err, const std::string& msg) {
    if (err) {
      throw std::runtime_error(msg + ": " + err.str());
    }
  }

  operator bool() const { return code_ != CUDA_SUCCESS; }

  CUresult code() const { return code_; }

  std::string str() const {
    const char* buf;
    CUresult err = cuGetErrorString(code_, &buf);
    if (err == CUDA_SUCCESS) {
      return buf;
    }
    return "Unknown";
  }

 private:
  CUresult code_;
};

}  // namespace cuda
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
