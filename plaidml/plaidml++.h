// Copyright 2018 Intel Corporation.
//
// This is the PlaidML C++ interface, which provides a higher level object
// oriented wrapper on top of the PlaidML C API.

#pragma once

#include <exception>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <half.hpp>

#include "plaidml/base/base_cpp.h"
#include "plaidml/plaidml.h"

using half_float::half;

namespace vertexai {
namespace plaidml {

// Import plaidml_datatype into the namespace
typedef plaidml_datatype datatype;

// Make a map for c++ types to PlaidML types
template <typename T>
struct to_plaidml_datatype {};
template <>
struct to_plaidml_datatype<int8_t> {
  static constexpr plaidml_datatype value = PLAIDML_DATA_INT8;
};
template <>
struct to_plaidml_datatype<int16_t> {
  static constexpr plaidml_datatype value = PLAIDML_DATA_INT16;
};
template <>
struct to_plaidml_datatype<int32_t> {
  static constexpr plaidml_datatype value = PLAIDML_DATA_INT32;
};
template <>
struct to_plaidml_datatype<int64_t> {
  static constexpr plaidml_datatype value = PLAIDML_DATA_INT64;
};
template <>
struct to_plaidml_datatype<uint8_t> {
  static constexpr plaidml_datatype value = PLAIDML_DATA_UINT8;
};
template <>
struct to_plaidml_datatype<uint16_t> {
  static constexpr plaidml_datatype value = PLAIDML_DATA_UINT16;
};
template <>
struct to_plaidml_datatype<uint32_t> {
  static constexpr plaidml_datatype value = PLAIDML_DATA_UINT32;
};
template <>
struct to_plaidml_datatype<uint64_t> {
  static constexpr plaidml_datatype value = PLAIDML_DATA_UINT64;
};
template <>
struct to_plaidml_datatype<half> {
  static constexpr plaidml_datatype value = PLAIDML_DATA_FLOAT16;
};
template <>
struct to_plaidml_datatype<float> {
  static constexpr plaidml_datatype value = PLAIDML_DATA_FLOAT32;
};
template <>
struct to_plaidml_datatype<double> {
  static constexpr plaidml_datatype value = PLAIDML_DATA_FLOAT64;
};

// Predeclare classes
class application;
class base_shape;
class base_tensor;
class buffer;
class compose;
class device;
class device_config;
class function;
class gradient;
class invoker;
template <typename T>
class mapping;
class placeholder;
template <typename T>
class shape;
template <typename T>
class tensor;
class variable;

// A dimension containing both size + stride
struct dimension {
  uint64_t size;
  int64_t stride;
};

inline std::vector<device_config> _enumerate_devices(const std::shared_ptr<ctx>& ctx,
                                                     std::shared_ptr<plaidml_device_enumerator> dev_enum);
inline std::vector<device_config> enumerate_devices(const std::shared_ptr<ctx>& ctx);
inline std::vector<device_config> enumerate_devices(const std::shared_ptr<ctx>& ctx, const std::string& config);

class base_shape {
  friend class base_tensor;
  friend class invoker;

 public:
  // Construct an empty shape with a specific data type
  explicit base_shape(const std::shared_ptr<ctx>& ctx, datatype dtype = PLAIDML_DATA_FLOAT32)
      : ctx_{ctx}, ptr_(plaidml_alloc_shape(ctx->get_ctx(), dtype), plaidml_free_shape) {
    vai_exception::check_and_throw(ptr_);
  }

  // Simple passthrough of C api for setup
  void add_dimension(size_t size, ptrdiff_t stride) {
    bool r = plaidml_add_dimension(ctx_->get_ctx(), ptr_.get(), size, stride);
    vai_exception::check_and_throw(r);
  }
  void set_offset(size_t offset) {
    bool r = plaidml_set_shape_offset(ctx_->get_ctx(), ptr_.get(), offset);
    vai_exception::check_and_throw(r);
  }

  // Add a dimension
  void push_back(const dimension& d) { add_dimension(d.size, d.stride); }

  // Add multiple dimensions
  template <typename L>
  void add_dimensions(const L& dims, ptrdiff_t initial_stride = 1) {
    ptrdiff_t stride = initial_stride;
    for (const auto& sz : dims) {
      stride *= sz;
    }
    for (const auto& sz : dims) {
      stride /= sz;
      add_dimension(sz, stride);
    }
  }

  // Make a simple shape with packed strides, last dimension lowest stride
  base_shape(const std::shared_ptr<ctx>& ctx, datatype dtype, const std::initializer_list<size_t>& il,
             uint64_t offset = 0)
      : base_shape(ctx, dtype) {
    add_dimensions(il);
  }

  // Get information about a shape
  datatype type() const { return plaidml_get_shape_type(ptr_.get()); }
  size_t dimensions() const { return plaidml_get_shape_dimension_count(ptr_.get()); }
  dimension operator[](size_t i) const { return dimension{size(i), stride(i)}; }
  uint64_t size(size_t i) const { return plaidml_get_shape_dimension_size(ptr_.get(), i); }
  int64_t stride(size_t i) const { return plaidml_get_shape_dimension_stride(ptr_.get(), i); }
  uint64_t buffer_size() const { return plaidml_get_shape_buffer_size(ptr_.get()); }

  const std::shared_ptr<ctx>& get_context() const { return ctx_; }

 protected:
  explicit base_shape(const std::shared_ptr<ctx>& ctx, const std::shared_ptr<plaidml_shape>& ptr)
      : ctx_{ctx}, ptr_{ptr} {}

  std::shared_ptr<ctx> ctx_;
  std::shared_ptr<plaidml_shape> ptr_;
};

template <typename T>
class shape : public base_shape {
 public:
  explicit shape(const std::shared_ptr<ctx>& ctx, datatype dt = to_plaidml_datatype<T>::value) : base_shape(ctx, dt) {}

  explicit shape(const base_shape& base, datatype dt = to_plaidml_datatype<T>::value) : base_shape(base) {
    if (dt != base.type()) {
      throw vai_exception(VAI_STATUS_INVALID_ARGUMENT, "Mismatched shape");
    }
  }

  shape(const std::shared_ptr<ctx>& ctx, const std::initializer_list<size_t>& il, uint64_t offset = 0)
      : base_shape(ctx, to_plaidml_datatype<T>::value, il, offset) {}
};

class buffer {
  friend class device;
  friend class base_tensor;

 public:
  buffer() {}

  void copy_into(const std::shared_ptr<ctx>& ctx, void* dst) {
    plaidml_mapping* mapping = plaidml_map_buffer_discard(ctx->get_ctx(), ptr_.get());
    char* src = plaidml_get_mapping_base(ctx->get_ctx(), mapping);
    size_t size = plaidml_get_mapping_size(ctx->get_ctx(), mapping);
    memcpy(dst, src, size);
    plaidml_free_mapping(mapping);
  }

  void copy_from(const std::shared_ptr<ctx>& ctx, const void* src) {
    plaidml_mapping* mapping = plaidml_map_buffer_current(ptr_.get(), nullptr, nullptr);
    char* dst = plaidml_get_mapping_base(ctx->get_ctx(), mapping);
    size_t size = plaidml_get_mapping_size(ctx->get_ctx(), mapping);
    memcpy(dst, src, size);
    plaidml_writeback_mapping(ctx->get_ctx(), mapping);
    plaidml_free_mapping(mapping);
  }

 private:
  std::shared_ptr<plaidml_buffer> ptr_;
  explicit buffer(const std::shared_ptr<plaidml_buffer>& ptr) : ptr_(ptr) {}
};

class base_tensor {
  friend class variable;

 public:
  base_tensor() {}
  base_tensor(const std::shared_ptr<ctx>& ctx, const buffer& buf, const base_shape& shape)
      : ctx_(ctx), buf_(buf.ptr_), shape_(shape.ptr_) {}

  base_shape get_shape() { return base_shape(ctx_, shape_); }
  buffer get_buffer() { return buffer(buf_); }
  std::shared_ptr<ctx> get_context() { return ctx_; }

 protected:
  std::shared_ptr<ctx> ctx_;
  std::shared_ptr<plaidml_buffer> buf_;
  std::shared_ptr<plaidml_shape> shape_;
};
// Indicates that the mapping will be used for reading the buffer.  The mapping
// will reflect the buffer's current contents.  By default, the implementation
// is free to either discard the mapping's contents or to write them back to
// the underlying buffer.
struct map_for_read_t {};
static constexpr map_for_read_t map_for_read = {};

// Indicates that the mapping will be used for writing to the buffer.  The
// mapping may not reflect the buffer's current contents; the implementation is
// free to construct the mapping with garbage data.  By default, the mapping's
// contents will be written back to the buffer when the mapping is deleted
// unless the deletion is due to an exceptional condition.
struct map_for_write_t {};
static constexpr map_for_write_t map_for_write = {};

// Indicates that the mapping will be used for read-write access to the buffer.
// The mapping will reflect the buffer's current contents.  By default, the
// mapping's contents will be written back to the buffer when the mapping is
// deleted unless the deletion is due to an exceptional condition.
struct map_for_update_t {};
static constexpr map_for_update_t map_for_update = {};

enum class mapping_destructor_behavior {
  writeback_if_normal,  // Write the contents on normal exits
  writeback_always,     // Always write the contents (including on exceptions)
  discard               // The implementation may discard the contents
};

template <typename T>
class mapping {
 public:
  // Construct an uninitialized mapping.
  mapping() {}

  ~mapping() { release(); }

  // Disallow copy + default construction
  mapping(const mapping& rhs) = delete;
  mapping& operator=(const mapping& rhs) = delete;

  // Allow moves
  mapping(mapping&& rhs)
      : ctx_{std::move(rhs.ctx_)},
        buf_{std::move(rhs.buf_)},
        sizes_{std::move(rhs.sizes_)},
        strides_{std::move(rhs.strides_)},
        map_{std::move(rhs.map_)},
        behavior_{std::move(rhs.behavior_)},
        mapped_{rhs.mapped_} {
    rhs.mapped_ = nullptr;
  }

  mapping& operator=(mapping&& rhs) {
    release();
    ctx_ = std::move(rhs.ctx_);
    buf_ = std::move(rhs.buf_);
    sizes_ = std::move(rhs.sizes_);
    strides_ = std::move(rhs.strides_);
    map_ = std::move(rhs.map_);
    behavior_ = std::move(rhs.behavior_);
    mapped_ = rhs.mapped_;
    rhs.mapped_ = nullptr;
    return *this;
  }

  // Provide access to raw buffer
  T* raw() { return mapped_; }

  // Explicitly set the destruction behavior.
  void set_destructor_behavior(mapping_destructor_behavior behavior) { behavior_ = behavior; }

  // Compute location of index, also do bounds check.  Note: this is a convience function;
  // it is not designed to be performant.
  T& at(const std::initializer_list<size_t>& idx) {
    if (idx.size() != sizes_.size()) {
      throw vai_exception(VAI_STATUS_OUT_OF_RANGE, "Invalid number of indexes in mapping access");
    }
    ptrdiff_t off = 0;
    for (size_t i = 0; i < sizes_.size(); i++) {
      if (*(idx.begin() + i) >= sizes_[i]) {
        throw vai_exception(VAI_STATUS_OUT_OF_RANGE, "Index out of bound on mapping access");
      }
      off += strides_[i] * *(idx.begin() + i);
    }
    return mapped_[off];
  }

  // Syntactic sugar
  template <typename... Args>
  T& operator()(Args... args) {
    return at({args...});
  }

 private:
  friend class tensor<T>;

  mapping(std::shared_ptr<ctx> ctx, std::shared_ptr<plaidml_buffer> buf, const std::shared_ptr<plaidml_shape>& shape,
          std::unique_ptr<plaidml_mapping> map, mapping_destructor_behavior behavior)
      : ctx_{std::move(ctx)}, buf_{std::move(buf)}, map_{std::move(map)}, behavior_{behavior} {
    sizes_.resize(plaidml_get_shape_dimension_count(shape.get()));
    strides_.resize(plaidml_get_shape_dimension_count(shape.get()));
    for (size_t i = 0; i < strides_.size(); i++) {
      sizes_[i] = plaidml_get_shape_dimension_size(shape.get(), i);
      strides_[i] = plaidml_get_shape_dimension_stride(shape.get(), i);
    }
    mapped_ = reinterpret_cast<T*>(plaidml_get_mapping_base(ctx_->get_ctx(), map_.get()));
    vai_exception::check_and_throw(mapped_);
  }

  void release() {
    if (!mapped_) {
      return;
    }
    switch (behavior_) {
      case mapping_destructor_behavior::writeback_if_normal:
#ifdef __cpp_lib_uncaught_exceptions
        if (std::uncaught_exceptions()) {
          break;
        }
#else
        if (std::uncaught_exception()) {
          break;
        }
#endif
      // fallthrough
      case mapping_destructor_behavior::writeback_always:
        plaidml_writeback_mapping(ctx_->get_ctx(), map_.get());
        break;
      case mapping_destructor_behavior::discard:
        break;
    }
    mapped_ = nullptr;
  }

  std::shared_ptr<ctx> ctx_;
  std::shared_ptr<plaidml_buffer> buf_;
  std::vector<size_t> sizes_;
  std::vector<ptrdiff_t> strides_;
  std::unique_ptr<plaidml_mapping> map_;
  mapping_destructor_behavior behavior_;
  T* mapped_ = nullptr;
};

template <typename T>
class tensor : public base_tensor {
  friend class mapping<T>;

 public:
  tensor() {}
  tensor(const std::shared_ptr<ctx>& ctx, const buffer& buf, const shape<T>& shape) : base_tensor(ctx, buf, shape) {}

  mapping<T> map(map_for_read_t) const {
    std::unique_ptr<plaidml_mapping> m{plaidml_map_buffer_current(buf_.get(), NULL, NULL)};
    return mapping<T>{ctx_, buf_, shape_, std::move(m), mapping_destructor_behavior::discard};
  }

  // Asynchronously creates a readable mapping.  The completion function should take a std::future<mapping<T>>,
  // which will be a ready future for the result of the mapping call.
  template <typename C>
  void map(map_for_read_t, vai_ctx* ctx, C&& on_complete) const {
    std::unique_ptr<completion> comp{
        static_cast<completion*>(new typed_completion<C>(ctx_, buf_, shape_, std::forward<C>(on_complete)))};
    plaidml_map_buffer_current(buf_.get(), &OnMapped, comp.release());
  }

  mapping<T> map(map_for_write_t) {
    std::unique_ptr<plaidml_mapping> m{plaidml_map_buffer_discard(ctx_->get_ctx(), buf_.get())};
    return mapping<T>{ctx_, buf_, shape_, std::move(m), mapping_destructor_behavior::writeback_if_normal};
  }

  mapping<T> map(map_for_update_t) {
    std::unique_ptr<plaidml_mapping> m{plaidml_map_buffer_current(buf_.get(), NULL, NULL)};
    return mapping<T>{ctx_, buf_, shape_, std::move(m), mapping_destructor_behavior::writeback_if_normal};
  }

 private:
  class completion {
   public:
    virtual ~completion() {}
    virtual void complete(plaidml_mapping* result) = 0;
  };

  template <typename C>
  class typed_completion final : public completion {
   public:
    typed_completion(std::shared_ptr<ctx> ctx, std::shared_ptr<plaidml_buffer> buf,
                     std::shared_ptr<plaidml_shape> shape, C&& on_complete)
        : ctx_{std::move(ctx)},
          buf_{std::move(buf)},
          shape_{std::move(shape)},
          on_complete_{std::forward<C>(on_complete)} {}

    void complete(plaidml_mapping* result) final {
      if (!result) {
        prom_.set_exception(vai_exception::current());
      } else {
        std::unique_ptr<plaidml_mapping> mp{result};
        prom_.set_value(mapping<T>{ctx_, std::move(buf_), shape_, std::move(mp), mapping_destructor_behavior::discard});
      }
      on_complete_(prom_.get_future());
    }

   private:
    std::shared_ptr<ctx> ctx_;
    std::shared_ptr<plaidml_buffer> buf_;
    std::shared_ptr<plaidml_shape> shape_;
    std::promise<mapping<T>> prom_;
    C on_complete_;
  };

  static void OnMapped(void* arg, plaidml_mapping* result) noexcept {
    std::unique_ptr<completion> comp{static_cast<completion*>(arg)};
    comp->complete(result);
  }
};

class placeholder {
  friend class variable;
  friend class compose;

 public:
  placeholder() {}
  explicit placeholder(size_t ndims) : ptr_(plaidml_alloc_placeholder(ndims), plaidml_free_var) {
    vai_exception::check_and_throw(ptr_);
  }

 private:
  std::shared_ptr<plaidml_var> ptr_;
};

class variable {
  friend class application;
  friend class compose;
  friend class function;
  friend class gradient;
  friend class invoker;

 public:
  variable() {}
  variable(const int64_t& val) : ptr_(plaidml_alloc_int64(val), plaidml_free_var) {  // NOLINT(runtime/explicit)
    vai_exception::check_and_throw(ptr_);
  }
  variable(const double& val) : ptr_(plaidml_alloc_real(val), plaidml_free_var) {  // NOLINT(runtime/explicit)
    vai_exception::check_and_throw(ptr_);
  }
  variable(const placeholder& val) : ptr_(val.ptr_) {}  // NOLINT(runtime/explicit)
  variable(const base_tensor& val)                      // NOLINT(runtime/explicit)
      : ptr_(plaidml_alloc_tensor(val.ctx_->get_ctx(), val.buf_.get(), val.shape_.get()), plaidml_free_var) {
    vai_exception::check_and_throw(ptr_);
  }

 private:
  std::shared_ptr<plaidml_var> ptr_;
};

class application {
  friend class function;
  friend class compose;

 public:
  application() {}

  operator variable() {
    if (plaidml_get_function_output_count(func_.get()) != 1) {
      throw std::runtime_error("Function application with non-unique return used in variable context");
    }
    return get_output(0);
  }

  variable get_output(size_t i) {
    if (i >= plaidml_get_function_output_count(func_.get())) {
      throw std::runtime_error("Attempting to get an invalid output index");
    }
    std::string oname = plaidml_get_function_output(func_.get(), i);
    return get_output(oname);
  }

  variable get_output(const std::string& name) {
    variable r;
    std::shared_ptr<plaidml_var> out(plaidml_apply_alloc_output(ptr_.get(), name.c_str()), plaidml_free_var);
    vai_exception::check_and_throw(out);
    r.ptr_ = out;
    return r;
  }

 private:
  std::shared_ptr<plaidml_function> func_;
  std::shared_ptr<plaidml_applier> ptr_;
  application(const std::shared_ptr<plaidml_function> func, const std::shared_ptr<plaidml_applier>& ptr)
      : func_(func), ptr_(ptr) {}
};

class function {
  friend class compose;
  friend class invoker;

 public:
  typedef std::vector<std::pair<std::string, variable>> parameters_t;
  typedef std::vector<variable> positional_t;

  // Invalid function
  function() {}

  // Make a function from code
  explicit function(const std::string& str, const std::string& id = "")
      : ptr_(plaidml_build_coded_function(str.c_str(), id.c_str()), plaidml_free_function) {
    vai_exception::check_and_throw(ptr_);
  }

  // Load and save function
  inline void load(const std::shared_ptr<ctx>& ctx, const device& dev,
                   const std::string& file);  // Later, after dev is defined
  void save(const std::string& file) {
    vai_exception::check_and_throw(plaidml_save_function(ptr_.get(), file.c_str()));
  }

  // Get information
  size_t num_inputs() { return plaidml_get_function_input_count(ptr_.get()); }
  size_t num_outputs() { return plaidml_get_function_output_count(ptr_.get()); }
  std::string input_name(size_t i) {
    const char* name = plaidml_get_function_input(ptr_.get(), i);
    return (name == NULL ? "" : name);
  }
  std::string output_name(size_t i) {
    const char* name = plaidml_get_function_output(ptr_.get(), i);
    return (name == NULL ? "" : name);
  }

  // Apply a function to values, produce new values, named parameters
  application apply(const parameters_t& inputs, const std::vector<application> prev = {}) {
    std::shared_ptr<plaidml_applier> app(plaidml_alloc_applier(ptr_.get()), plaidml_free_applier);
    vai_exception::check_and_throw(app);
    for (const auto& papp : prev) {
      bool r = plaidml_apply_add_dependency(app.get(), papp.ptr_.get());
      vai_exception::check_and_throw(r);
    }
    for (const auto& arg : inputs) {
      bool r = plaidml_apply_add_input(app.get(), arg.first.c_str(), arg.second.ptr_.get());
      vai_exception::check_and_throw(r);
    }
    return application(ptr_, app);
  }

  application apply(const positional_t& inputs, const std::vector<application> prev = {}) {
    if (inputs.size() != num_inputs()) {
      throw std::runtime_error("Mismatched number of input in application: " + std::to_string(inputs.size()) + " vs " +
                               std::to_string(num_inputs()));
    }
    std::shared_ptr<plaidml_applier> app(plaidml_alloc_applier(ptr_.get()), plaidml_free_applier);
    vai_exception::check_and_throw(app);
    for (const auto& papp : prev) {
      bool r = plaidml_apply_add_dependency(app.get(), papp.ptr_.get());
      vai_exception::check_and_throw(r);
    }
    for (size_t idx = 0; idx < inputs.size(); ++idx) {
      bool r = plaidml_apply_add_input(app.get(), input_name(idx).c_str(), inputs[idx].ptr_.get());
      vai_exception::check_and_throw(r);
    }
    return application(ptr_, app);
  }

  // Operator() for ease of use in apply
  template <typename... Params>
  application operator()(Params... params) {
    return apply(std::vector<variable>{params...});
  }

 private:
  std::shared_ptr<plaidml_function> ptr_;
  explicit function(const std::shared_ptr<plaidml_function>& ptr) : ptr_(ptr) {}
};

class compose {
 public:
  explicit compose(const std::string name = "") : ptr_(plaidml_alloc_composer(), plaidml_free_composer) {
    vai_exception::check_and_throw(ptr_);
  }

  compose& input(const std::string& name, const placeholder& p) {
    bool r = plaidml_add_composer_input(ptr_.get(), name.c_str(), p.ptr_.get());
    vai_exception::check_and_throw(r);
    return *this;
  }

  compose& output(const std::string& name, const variable& p) {
    bool r = plaidml_add_composer_output(ptr_.get(), name.c_str(), p.ptr_.get());
    vai_exception::check_and_throw(r);
    return *this;
  }

  compose& dependency(const application& prev) {
    bool r = plaidml_add_composer_dependency(ptr_.get(), prev.ptr_.get());
    vai_exception::check_and_throw(r);
    return *this;
  }

  compose& update(const base_tensor& lhs, const variable& rhs) {
    variable tvar = lhs;
    bool r = plaidml_add_composer_update(ptr_.get(), tvar.ptr_.get(), rhs.ptr_.get());
    vai_exception::check_and_throw(r);
    return *this;
  }

  operator function() {
    std::shared_ptr<plaidml_function> func(plaidml_build_composed_function(ptr_.get()), plaidml_free_function);
    vai_exception::check_and_throw(func);
    return function(func);
  }

 private:
  std::shared_ptr<plaidml_composer> ptr_;
};

class invoker {
 public:
  invoker() {}
  invoker(const invoker&) = delete;
  invoker(invoker&&) = default;
  invoker& operator=(const invoker&) = delete;
  invoker& operator=(invoker&&) = default;

  invoker(const std::shared_ptr<ctx>& ctx, const function& func)
      : ctx_{ctx}, invoker_{plaidml_alloc_invoker(ctx_->get_ctx(), func.ptr_.get())} {
    vai_exception::check_and_throw(invoker_);
  }

  invoker& set_input(const std::string& name, const variable& var) {
    auto r = plaidml_set_invoker_input(invoker_.get(), name.c_str(), var.ptr_.get());
    vai_exception::check_and_throw(r);
    return *this;
  }

  invoker& set_output(const std::string& name, const variable& var) {
    auto r = plaidml_set_invoker_output(invoker_.get(), name.c_str(), var.ptr_.get());
    vai_exception::check_and_throw(r);
    return *this;
  }

  base_shape output_shape(const std::string& name) {
    std::shared_ptr<plaidml_shape> shp{plaidml_alloc_invoker_output_shape(invoker_.get(), name.c_str()),
                                       plaidml_free_shape};
    vai_exception::check_and_throw(shp);
    return base_shape{ctx_, std::move(shp)};
  }

  void save(const std::string& file, plaidml_file_format format) {
    vai_exception::check_and_throw(plaidml_save_invoker(invoker_.get(), file.c_str(), format));
  }

  void set_const() { vai_exception::check_and_throw(plaidml_set_invoker_const(invoker_.get())); }

  std::unique_ptr<plaidml_invocation> invoke() {
    std::unique_ptr<plaidml_invocation> invocation{plaidml_schedule_invocation(ctx_->get_ctx(), invoker_.get())};
    vai_exception::check_and_throw(invocation);
    return invocation;
  }

  std::unique_ptr<plaidml_invocation> invoke(const std::shared_ptr<ctx>& ctx) {
    std::unique_ptr<plaidml_invocation> invocation{plaidml_schedule_invocation(ctx->get_ctx(), invoker_.get())};
    vai_exception::check_and_throw(invocation);
    return invocation;
  }

 private:
  std::shared_ptr<ctx> ctx_;
  std::unique_ptr<plaidml_invoker> invoker_;
};

// TODO: Fix this!
class gradient {
 public:
  explicit gradient(const variable& var) : ptr_(plaidml_alloc_gradient(var.ptr_.get()), plaidml_free_gradient) {
    vai_exception::check_and_throw(ptr_);
  }
  variable operator()(const variable& v) {
    variable r;
    plaidml_var* var = plaidml_compute_grad_wrt(ptr_.get(), v.ptr_.get());
    vai_exception::check_and_throw(var);
    r.ptr_ = std::shared_ptr<plaidml_var>(var, plaidml_free_var);
    return r;
  }

 private:
  std::shared_ptr<plaidml_gradient> ptr_;
};

class device {
  friend class function;
  friend class device_config;

 public:
  device() = default;

  bool operator!() const { return !ptr_; }

  buffer allocate(uint64_t size) const {
    buffer r;

    r.ptr_ =
        std::shared_ptr<plaidml_buffer>(plaidml_alloc_buffer(ctx_->get_ctx(), ptr_.get(), size), plaidml_free_buffer);
    vai_exception::check_and_throw(r.ptr_);
    return r;
  }

  base_tensor allocate(const base_shape& s) const { return base_tensor(s.get_context(), allocate(s.buffer_size()), s); }

  template <class T>
  tensor<T> allocate(const shape<T>& s) const {
    return tensor<T>(s.get_context(), allocate(s.buffer_size()), s);
  }

 private:
  explicit device(const std::shared_ptr<ctx>& ctx, plaidml_device* raw) : ctx_{ctx}, ptr_(raw, plaidml_close_device) {}
  std::shared_ptr<ctx> ctx_;
  std::shared_ptr<plaidml_device> ptr_;
  const std::shared_ptr<ctx>& get_context() const { return ctx_; }
};

class device_config {
  friend std::vector<device_config> _enumerate_devices(const std::shared_ptr<ctx>& ctx,
                                                       std::shared_ptr<plaidml_device_enumerator> dev_enum);

 public:
  // Get any string based property
  std::string get_string_prop(plaidml_device_property prop) const {
    size_t out_size;
    bool r = plaidml_query_devconf(ctx_->get_ctx(), config_, prop, NULL, 0, &out_size);
    vai_exception::check_and_throw(r);
    std::string str(out_size, '\0');
    r = plaidml_query_devconf(ctx_->get_ctx(), config_, prop, &str[0], str.size(), NULL);
    str.pop_back();
    vai_exception::check_and_throw(r);
    return str;
  }

  // Convenience functions for current properties
  std::string id() const { return get_string_prop(PLAIDML_DEVICE_ID); }
  std::string config() const { return get_string_prop(PLAIDML_DEVICE_CONFIG); }
  std::string description() const { return get_string_prop(PLAIDML_DEVICE_DESCRIPTION); }
  std::string details() const { return get_string_prop(PLAIDML_DEVICE_DETAILS); }

  // Open the device
  device open() const {
    device dev(ctx_, plaidml_open_device(ctx_->get_ctx(), config_));
    vai_exception::check_and_throw(dev.ptr_);
    return dev;
  }

 private:
  device_config(const std::shared_ptr<ctx>& ctx, const std::shared_ptr<plaidml_device_enumerator>& dev_enum,
                plaidml_devconf* config)
      : ctx_(ctx), dev_enum_(dev_enum), config_(config) {}

  std::shared_ptr<ctx> ctx_;
  std::shared_ptr<plaidml_device_enumerator> dev_enum_;
  plaidml_devconf* config_;
};

std::vector<device_config> _enumerate_devices(const std::shared_ptr<ctx>& ctx,
                                              std::shared_ptr<plaidml_device_enumerator> dev_enum) {
  std::vector<device_config> out;
  size_t i = 0;
  while (true) {
    plaidml_devconf* conf = plaidml_get_devconf(ctx->get_ctx(), dev_enum.get(), i);
    if (conf == NULL) break;
    i++;
    out.push_back(device_config(ctx, dev_enum, conf));
  }
  vai_clear_status();  // Since we always walk off the list, clear errors
  return out;
}

std::vector<device_config> enumerate_devices(const std::shared_ptr<ctx>& ctx) {
  std::shared_ptr<plaidml_device_enumerator> dev_enum(plaidml_alloc_device_enumerator(ctx->get_ctx(), NULL, NULL),
                                                      plaidml_free_device_enumerator);
  vai_exception::check_and_throw(dev_enum);
  return _enumerate_devices(ctx, dev_enum);
}

std::vector<device_config> enumerate_devices(const std::shared_ptr<ctx>& ctx, const std::string& config) {
  std::vector<device_config> out;
  std::shared_ptr<plaidml_device_enumerator> dev_enum(
      plaidml_alloc_device_enumerator_with_config(ctx->get_ctx(), config.c_str(), NULL, NULL),
      plaidml_free_device_enumerator);
  vai_exception::check_and_throw(dev_enum);
  return _enumerate_devices(ctx, dev_enum);
}

// Actually needs definitions of both classes
void function::load(const std::shared_ptr<ctx>& ctx, const device& dev, const std::string& file) {
  ptr_ = std::shared_ptr<plaidml_function>(plaidml_load_function(ctx->get_ctx(), dev.ptr_.get(), file.c_str()),
                                           plaidml_free_function);
  vai_exception::check_and_throw(ptr_);
}

}  // namespace plaidml
}  // namespace vertexai
