// Copyright 2019 Intel Corporation.

#pragma once

#include <cstring>
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

inline std::shared_ptr<plaidml_buffer> make_plaidml_buffer(plaidml_buffer* ptr) {
  return std::shared_ptr<plaidml_buffer>(ptr, Deleter<plaidml_buffer>{plaidml_buffer_free});
}

inline std::shared_ptr<plaidml_view> make_plaidml_view(plaidml_view* ptr) {
  return std::shared_ptr<plaidml_view>(ptr, Deleter<plaidml_view>{plaidml_view_free});
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

class View {
  friend class Buffer;

 public:
  char* data() {  //
    return ffi::call<char*>(plaidml_view_data, ptr_.get());
  }

  size_t size() {  //
    return ffi::call<size_t>(plaidml_view_size, ptr_.get());
  }

  void writeback() {  //
    ffi::call_void(plaidml_view_writeback, ptr_.get());
  }

 private:
  explicit View(const std::shared_ptr<plaidml_view>& ptr) : ptr_(ptr) {}

 private:
  std::shared_ptr<plaidml_view> ptr_;
};

class Buffer {
 public:
  Buffer() = default;
  Buffer(const std::string& device_id, const TensorShape& shape)
      : ptr_(details::make_plaidml_buffer(
            ffi::call<plaidml_buffer*>(plaidml_buffer_alloc, device_id.c_str(), shape.nbytes()))),
        shape_(shape) {}

  plaidml_buffer* as_ptr() const {  //
    return ptr_.get();
  }

  View mmap_current() {
    return View(details::make_plaidml_view(ffi::call<plaidml_view*>(plaidml_buffer_mmap_current, ptr_.get())));
  }

  View mmap_discard() {
    return View(details::make_plaidml_view(ffi::call<plaidml_view*>(plaidml_buffer_mmap_discard, ptr_.get())));
  }

  void copy_into(void* dst) {
    auto view = mmap_current();
    memcpy(dst, view.data(), view.size());
  }

  void copy_from(const void* src) {
    auto view = mmap_discard();
    memcpy(view.data(), src, view.size());
    view.writeback();
  }

 private:
  std::shared_ptr<plaidml_buffer> ptr_;
  TensorShape shape_;
};

struct Settings {
  static std::string get(const std::string& key) {
    return ffi::str(ffi::call<plaidml_string*>(plaidml_settings_get, key.c_str()));
  }

  static void set(const std::string& key, const std::string& value) {
    ffi::call_void(plaidml_settings_set, key.c_str(), value.c_str());
  }
};

}  // namespace plaidml
