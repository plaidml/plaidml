// Copyright 2019 Intel Corporation.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "plaidml2/core/ffi.h"

namespace plaidml {

namespace ffi {

inline std::string str(plaidml_string* ptr) {
  std::string ret{plaidml_string_ptr(ptr)};
  plaidml_string_free(ptr);
  return ret;
}

template <typename T, typename F, typename... Args>
T call(F fn, Args... args) {
  plaidml_error err;
  auto ret = fn(&err, args...);
  if (err.code) {
    throw std::runtime_error(str(err.msg));
  }
  return ret;
}

template <typename F, typename... Args>
void call_void(F fn, Args... args) {
  plaidml_error err;
  fn(&err, args...);
  if (err.code) {
    throw std::runtime_error(str(err.msg));
  }
}

}  // namespace ffi

namespace details {

template <typename T>
struct Deleter {
  std::function<void(plaidml_error*, T*)> fn;
  void operator()(T* ptr) { ffi::call_void(fn, ptr); }
};

inline std::shared_ptr<plaidml_shape> make_plaidml_shape(plaidml_shape* ptr) {
  return std::shared_ptr<plaidml_shape>(ptr, Deleter<plaidml_shape>{plaidml_shape_free});
}

}  // namespace details

inline void init() {  //
  ffi::call_void(plaidml_init);
}

class TensorShape {
 public:
  TensorShape()
      : ptr_(details::make_plaidml_shape(
            ffi::call<plaidml_shape*>(plaidml_shape_alloc, PLAIDML_DATA_INVALID, 0, nullptr, nullptr))) {}

  TensorShape(plaidml_datatype dtype,  //
              const std::vector<int64_t>& sizes) {
    size_t stride = 1;
    std::vector<int64_t> strides(sizes.size());
    for (int i = sizes.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= sizes[i];
    }
    ptr_ = details::make_plaidml_shape(
        ffi::call<plaidml_shape*>(plaidml_shape_alloc, dtype, sizes.size(), sizes.data(), strides.data()));
  }

  TensorShape(plaidml_datatype dtype,             //
              const std::vector<int64_t>& sizes,  //
              const std::vector<int64_t>& strides) {
    if (sizes.size() != strides.size()) {
      throw std::runtime_error("Sizes and strides must have the same rank.");
    }
    ptr_ = details::make_plaidml_shape(
        ffi::call<plaidml_shape*>(plaidml_shape_alloc, dtype, sizes.size(), sizes.data(), strides.data()));
  }

  explicit TensorShape(const std::shared_ptr<plaidml_shape>& ptr) : ptr_(ptr) {}

  plaidml_datatype dtype() const { return ffi::call<plaidml_datatype>(plaidml_shape_get_dtype, ptr_.get()); }
  size_t ndims() const { return ffi::call<size_t>(plaidml_shape_get_ndims, ptr_.get()); }
  uint64_t nbytes() const { return ffi::call<uint64_t>(plaidml_shape_get_nbytes, ptr_.get()); }
  std::string str() const { return ffi::str(ffi::call<plaidml_string*>(plaidml_shape_repr, ptr_.get())); }
  bool operator==(const TensorShape& rhs) const { return str() == rhs.str(); }
  plaidml_shape* as_ptr() const { return ptr_.get(); }

 private:
  std::shared_ptr<plaidml_shape> ptr_;
};

}  // namespace plaidml
