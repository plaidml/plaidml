// Copyright 2017-2018 Intel Corporation.

#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

#include <string>

#include "base/util/logging.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

// A little wrapper class to simplify handling OpenCL errors.
class Err final {
 public:
  // A simple Check wrapper, for common error handling situations (i.e. fixed
  // string description).  This will throw an exception with a useful message
  // iff the supplied Err value wraps something other than CL_SUCCESS.
  static void Check(Err err, const std::string& msg) {
    if (err) {
      std::rethrow_exception(err.ToException(msg));
    }
  }

  Err() : code_(CL_SUCCESS) {}

  // clang-format off
  //cppcheck-suppress noExplicitConstructor  // NOLINT
  Err(cl_int code);  // NOLINT
  // clang-format on

  // Errors are "true" if there is an error.  So the idiom is:
  // Err err = /* something */
  // if (err) {
  //   /* throw something */
  // }
  operator bool() const { return code_ != CL_SUCCESS; }

  cl_int* ptr() { return &code_; }
  const char* str() const;
  cl_int code() const { return code_; }

  std::exception_ptr ToException(const std::string& msg) const {
    if (code_ == CL_SUCCESS) {
      return std::exception_ptr();
    }
    std::string err_msg = msg + ": " + str();
    LOG(ERROR) << err_msg;
    return std::make_exception_ptr(std::runtime_error(err_msg));
  }

 private:
  cl_int code_;
};

inline std::ostream& operator<<(std::ostream& o, const Err& err) { return o << err.str(); }

// Retain/Release overloads for OpenCL objects.
inline void Retain(cl_event e) {
  Err err = clRetainEvent(e);
  LOG_IF(err, ERROR) << "clRetainEvent: " << err.str();
}

inline void Release(cl_event e) {
  Err err = clReleaseEvent(e);
  LOG_IF(err, ERROR) << "clReleaseEvent: " << err.str();
}

inline void Retain(cl_mem m) {
  Err err = clRetainMemObject(m);
  LOG_IF(err, ERROR) << "clRetainMemObject: " << err.str();
}

inline void Release(cl_mem m) {
  Err err = clReleaseMemObject(m);
  LOG_IF(err, ERROR) << "clReleaseMemObject: " << err.str();
}

inline void Retain(cl_kernel k) {
  Err err = clRetainKernel(k);
  LOG_IF(err, ERROR) << "clRetainKernel: " << err.str();
}

inline void Release(cl_kernel k) {
  Err err = clReleaseKernel(k);
  LOG_IF(err, ERROR) << "clReleaseKernel: " << err.str();
}

inline void Retain(cl_context c) {
  Err err = clRetainContext(c);
  LOG_IF(err, ERROR) << "clRetainContext: " << err.str();
}

inline void Release(cl_context c) {
  Err err = clReleaseContext(c);
  LOG_IF(err, ERROR) << "clReleaseContext: " << err.str();
}

inline void Retain(cl_program p) {
  Err err = clRetainProgram(p);
  LOG_IF(err, ERROR) << "clRetainProgram: " << err.str();
}

inline void Release(cl_program p) {
  Err err = clReleaseProgram(p);
  LOG_IF(err, ERROR) << "clReleaseProgram: " << err.str();
}

inline void Retain(cl_command_queue c) {
  Err err = clRetainCommandQueue(c);
  LOG_IF(err, ERROR) << "clRetainCommandQueue: " << err.str();
}

inline void Release(cl_command_queue c) {
  Err err = clReleaseCommandQueue(c);
  LOG_IF(err, ERROR) << "clReleaseCommandQueue: " << err.str();
}

// A simple wrapper for OpenCL objects, providing RAII support.  If a wrapper
// holds an object, the wrapper retains a refcount on that object.  If the
// wrapper is assigned a bare OpenCL object, the wrapper assumes the object's
// refcount; this allows the wrapper to be used as an lvalue accepting objects
// returned by OpenCL library calls.
template <typename O>
class CLObj final {
 public:
  CLObj() {}

  // clang-format off
  //cppcheck-suppress noExplicitConstructor  // NOLINT
  CLObj(O obj) : obj_{obj} {}  // NOLINT
  // clang-format on

  CLObj(const CLObj<O>& other) : obj_{other.obj_} {
    if (obj_) {
      Retain(obj_);
    }
  }

  CLObj(CLObj<O>&& other) : obj_{other.obj_} { other.obj_ = nullptr; }

  CLObj& operator=(const CLObj<O>& other) {
    if (other.obj_) {
      Retain(other.obj_);
    }
    if (obj_) {
      Release(obj_);
    }
    obj_ = other.obj_;
    return *this;
  }

  CLObj& operator=(CLObj<O>&& other) {
    if (obj_) {
      Release(obj_);
    }
    obj_ = other.obj_;
    other.obj_ = nullptr;
    return *this;
  }

  ~CLObj() {
    if (obj_) {
      Release(obj_);
    }
  }

  void reset() {
    if (obj_) {
      Release(obj_);
      obj_ = nullptr;
    }
  }

  // Returns the object held by this wrapper, without retaining it or releasing
  // it.  Do not create a new wrapper from the result without explicitly
  // retaining it.
  O get() const { return obj_; }

  operator bool() const { return obj_ != nullptr; }

  bool operator==(const CLObj<O>& other) const { return obj_ == other.obj_; }

  bool operator==(O other) const { return obj_ == other; }

  bool operator!=(const CLObj<O>& other) const { return obj_ != other.obj_; }

  bool operator!=(O other) const { return obj_ != other; }

  // Releases the object held by the wrapper (if any), and returns a pointer
  // to the newly zero object within the wrapper.  This is intended for use
  // with OpenCL calls that use output parameters to return objects.
  O* LvaluePtr() {
    if (obj_) {
      Release(obj_);
    }
    obj_ = nullptr;
    return &obj_;
  }

 private:
  O obj_ = nullptr;
};

template <typename O>
inline bool operator<(const CLObj<O>& lhs, const CLObj<O>& rhs) {
  return lhs.get() < rhs.get();
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
