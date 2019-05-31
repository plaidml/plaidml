// Copyright 2019 Intel Corporation.

#include "plaidml/edsl/edsl.h"

#include <algorithm>
#include <iterator>
#include <sstream>

#include "plaidml/edsl/ffi.h"

namespace vertexai {
namespace plaidml {
namespace edsl {

struct IndexedTensor::Impl {
  struct ComboParts {
    tile_combo_op op;
    std::vector<tile_expr*> args;
  };

  std::shared_ptr<tile_expr> unary;
  std::shared_ptr<ComboParts> nary;
  Tensor::Impl* src = nullptr;
  void MakeContraction(tile_agg_op agg_op, const IndexedTensor& rhs);
};

struct Program::Impl {
  std::shared_ptr<tile_program> ptr;
};

struct TensorDim::Impl {
  std::unique_ptr<size_t> size;
};

struct TensorIndex::Impl {
  std::shared_ptr<tile_poly_expr> ptr;
};

struct TensorShape::Impl {
  std::shared_ptr<tile_shape> ptr;
};

struct Tensor::Impl {
  std::shared_ptr<tile_expr> ptr;
  std::vector<TensorDim> dims;
  std::string name;
};

namespace {

std::string ffi_str(tile_string* ptr) {
  std::string ret{tile_string_ptr(ptr)};
  tile_string_free(ptr);
  return ret;
}

template <typename T, typename F, typename... Args>
T ffi_call(F fn, Args... args) {
  tile_error err;
  auto ret = fn(&err, args...);
  if (err.code) {
    throw std::runtime_error(ffi_str(err.msg));
  }
  return ret;
}

template <typename F, typename... Args>
void ffi_call_void(F fn, Args... args) {
  tile_error err;
  fn(&err, args...);
  if (err.code) {
    throw std::runtime_error(ffi_str(err.msg));
  }
}

template <typename T>
struct Deleter {
  std::function<void(tile_error*, T*)> fn;
  void operator()(T* ptr) { ffi_call_void(fn, ptr); }
};

}  // namespace

std::shared_ptr<tile_shape> make_tile_shape(tile_shape* ptr) {
  return std::shared_ptr<tile_shape>(ptr, Deleter<tile_shape>{tile_shape_free});
}

std::shared_ptr<tile_expr> make_tile_expr(tile_expr* ptr) {
  return std::shared_ptr<tile_expr>(ptr, Deleter<tile_expr>{tile_expr_free});
}

std::shared_ptr<tile_poly_expr> make_tile_poly_expr(tile_poly_expr* ptr) {
  return std::shared_ptr<tile_poly_expr>(ptr, Deleter<tile_poly_expr>{tile_poly_expr_free});
}

std::shared_ptr<tile_program> make_tile_program(tile_program* ptr) {
  return std::shared_ptr<tile_program>(ptr, Deleter<tile_program>{tile_program_free});
}

class TensorFriend {
 public:
  static tile_program* evaluate(const std::string& name, const std::vector<Tensor>& tensors) {
    std::vector<tile_expr*> exprs;
    for (const auto& tensor : tensors) {
      exprs.emplace_back(tensor.impl_->ptr.get());
    }
    return ffi_call<tile_program*>(tile_program_evaluate, name.c_str(), exprs.size(), exprs.data());
  }

  static TensorIndex MakePolyOp(tile_poly_op op, const std::vector<TensorIndex>& args) {
    std::vector<tile_poly_expr*> operands;
    for (const auto& arg : args) {
      operands.push_back(arg.impl_->ptr.get());
    }
    auto impl = std::make_unique<TensorIndex::Impl>();
    impl->ptr = make_tile_poly_expr(  //
        ffi_call<tile_poly_expr*>(    //
            tile_poly_expr_op,        //
            op,                       //
            operands.size(),          //
            operands.data()));
    return TensorIndex{std::move(impl)};
  }

  static TensorIndex MakeMixedPolyBinaryOp(tile_poly_op op, const TensorIndex& idx, const TensorDim& dim,
                                           bool lhs_first) {
    if (!dim.impl_->size) {
      throw std::runtime_error("Undefined dimension.");
    }
    std::vector<tile_poly_expr*> operands;
    auto dim_ptr = ffi_call<tile_poly_expr*>(tile_poly_expr_literal, *dim.impl_->size);
    if (lhs_first) {
      operands.emplace_back(idx.impl_->ptr.get());
      operands.emplace_back(dim_ptr);
    } else {
      operands.emplace_back(dim_ptr);
      operands.emplace_back(idx.impl_->ptr.get());
    }
    auto impl = std::make_unique<TensorIndex::Impl>();
    impl->ptr = make_tile_poly_expr(  //
        ffi_call<tile_poly_expr*>(    //
            tile_poly_expr_op,        //
            op,                       //
            operands.size(),          //
            operands.data()));
    return TensorIndex{std::move(impl)};
  }

  static TensorDim DimOp(tile_poly_op op, const std::vector<TensorDim>& args) {
    std::vector<size_t> sizes;
    for (const auto& arg : args) {
      if (!arg.impl_->size) {
        throw std::runtime_error("Undefined dimension.");
      }
      sizes.emplace_back(*arg.impl_->size);
    }
    switch (op) {
      case TILE_POLY_OP_NEG:
        return TensorDim{-sizes[0]};
      case TILE_POLY_OP_ADD:
        return TensorDim{sizes[0] + sizes[1]};
      case TILE_POLY_OP_SUB:
        return TensorDim{sizes[0] - sizes[1]};
      case TILE_POLY_OP_MUL:
        return TensorDim{sizes[0] * sizes[1]};
      case TILE_POLY_OP_DIV:
        return TensorDim{sizes[0] / sizes[1]};
      default:
        throw std::runtime_error("Invalid poly op");
    }
  }

  inline static const IndexedTensor::Impl* GetImpl(const IndexedTensor& tensor) { return tensor.impl_.get(); }

  inline static const TensorIndex::Impl* GetImpl(const TensorIndex& idx) { return idx.impl_.get(); }

  inline static const Tensor::Impl* GetImpl(const Tensor& tensor) { return tensor.impl_.get(); }

  inline static const TensorShape::Impl* GetImpl(const TensorShape& shape) { return shape.impl_.get(); }

  static IndexedTensor ComboParts(tile_combo_op op, const std::vector<const IndexedTensor*>& args) {
    auto impl = std::make_unique<IndexedTensor::Impl>();
    impl->nary.reset(new IndexedTensor::Impl::ComboParts);
    impl->nary->op = op;
    for (const auto& arg : args) {
      impl->nary->args.emplace_back(arg->impl_->unary.get());
    }
    return IndexedTensor{std::move(impl)};
  }

  static Tensor Call(const std::string& fn, const std::vector<Tensor>& args) {
    std::vector<tile_expr*> ptrs(args.size());
    for (size_t i = 0; i < args.size(); i++) {
      ptrs[i] = args[i].impl_->ptr.get();
    }
    auto impl = std::make_unique<Tensor::Impl>();
    impl->ptr = make_tile_expr(  //
        ffi_call<tile_expr*>(    //
            tile_expr_call,      //
            fn.c_str(),          //
            ptrs.size(),         //
            ptrs.data()));
    return Tensor{std::move(impl)};
  }
};

TensorShape::TensorShape(const std::shared_ptr<Impl>& impl) : impl_(impl) {}

TensorShape::TensorShape(plaidml_datatype dtype,              //
                         const std::vector<uint64_t>& sizes,  //
                         const std::string& layout)
    : impl_(new Impl) {
  impl_->ptr = make_tile_shape(ffi_call<tile_shape*>(tile_shape_alloc, dtype, layout.c_str()));
  size_t stride = 1;
  std::vector<int64_t> strides(sizes.size());
  for (int i = sizes.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= sizes[i];
  }
  for (size_t i = 0; i < sizes.size(); i++) {
    ffi_call_void(tile_shape_add_dimension, impl_->ptr.get(), sizes[i], strides[i]);
  }
}

TensorShape::TensorShape(plaidml_datatype dtype,               //
                         const std::vector<uint64_t>& sizes,   //
                         const std::vector<int64_t>& strides,  //
                         const std::string& layout)
    : impl_(new Impl) {
  impl_->ptr = make_tile_shape(ffi_call<tile_shape*>(tile_shape_alloc, dtype, layout.c_str()));
  if (sizes.size() != strides.size()) {
    throw std::runtime_error("Sizes and strides must have same rank.");
  }
  for (size_t i = 0; i < sizes.size(); i++) {
    ffi_call_void(tile_shape_add_dimension, impl_->ptr.get(), sizes[i], strides[i]);
  }
}

plaidml_datatype TensorShape::type() const { return ffi_call<plaidml_datatype>(tile_shape_get_type, impl_->ptr.get()); }

size_t TensorShape::rank() const { return ffi_call<size_t>(tile_shape_get_rank, impl_->ptr.get()); }

uint64_t TensorShape::size_at(size_t dim) const {
  return ffi_call<uint64_t>(tile_shape_get_dimension_size, impl_->ptr.get(), dim);
}

int64_t TensorShape::stride_at(size_t dim) const {
  return ffi_call<int64_t>(tile_shape_get_dimension_stride, impl_->ptr.get(), dim);
}

uint64_t TensorShape::byte_size() const { return ffi_call<uint64_t>(tile_shape_get_byte_size, impl_->ptr.get()); }

const void* TensorShape::ptr() const { return ffi_call<const void*>(tile_shape_get_ptr, impl_->ptr.get()); }

std::string TensorShape::str() const { return ffi_str(ffi_call<tile_string*>(tile_shape_repr, impl_->ptr.get())); }

bool TensorShape::operator==(const TensorShape& rhs) const { return str() == rhs.str(); }

Program::Program(const std::string& name, const std::vector<Tensor>& tensors) : impl_(new Impl) {
  impl_->ptr = make_tile_program(TensorFriend::evaluate(name, tensors));
}

std::string Program::str() const { return ffi_str(ffi_call<tile_string*>(tile_program_repr, impl_->ptr.get())); }

const void* Program::runinfo() const { return ffi_call<const void*>(tile_program_runinfo, impl_->ptr.get()); }

TensorIndex::TensorIndex() : impl_(std::make_shared<Impl>()) {
  impl_->ptr = make_tile_poly_expr(  //
      ffi_call<tile_poly_expr*>(     //
          tile_poly_expr_index,      //
          ""));
}

TensorIndex::TensorIndex(size_t value) : impl_(std::make_shared<Impl>()) {
  impl_->ptr = make_tile_poly_expr(  //
      ffi_call<tile_poly_expr*>(     //
          tile_poly_expr_literal,    //
          value));
}

TensorIndex::TensorIndex(const std::string& name) : impl_(std::make_shared<Impl>()) {
  impl_->ptr = make_tile_poly_expr(  //
      ffi_call<tile_poly_expr*>(     //
          tile_poly_expr_index,      //
          name.c_str()));
}

TensorIndex::TensorIndex(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

TensorIndex TensorIndex::operator-() const { return TensorFriend::MakePolyOp(TILE_POLY_OP_NEG, {*this}); }

#define TILE_CC_DEFINE_TENSOR_IDXDIM_BINARY_OPS(_op_, _poly_op_)              \
  TensorIndex operator _op_(const TensorIndex& lhs, const TensorIndex& rhs) { \
    return TensorFriend::MakePolyOp(_poly_op_, {lhs, rhs});                   \
  }                                                                           \
  TensorIndex operator _op_(const TensorIndex& lhs, size_t rhs) {             \
    return TensorFriend::MakePolyOp(_poly_op_, {lhs, TensorIndex{rhs}});      \
  }                                                                           \
  TensorIndex operator _op_(size_t lhs, const TensorIndex& rhs) {             \
    return TensorFriend::MakePolyOp(_poly_op_, {TensorIndex{lhs}, rhs});      \
  }                                                                           \
  TensorIndex operator _op_(const TensorIndex& lhs, const TensorDim& rhs) {   \
    return TensorFriend::MakeMixedPolyBinaryOp(_poly_op_, lhs, rhs, true);    \
  }                                                                           \
  TensorIndex operator _op_(const TensorDim& lhs, const TensorIndex& rhs) {   \
    return TensorFriend::MakeMixedPolyBinaryOp(_poly_op_, rhs, lhs, false);   \
  }                                                                           \
  TensorDim operator _op_(const TensorDim& lhs, const TensorDim& rhs) {       \
    return TensorFriend::DimOp(_poly_op_, {lhs, rhs});                        \
  }                                                                           \
  TensorDim operator _op_(size_t lhs, const TensorDim& rhs) {                 \
    return TensorFriend::DimOp(_poly_op_, {TensorDim{lhs}, rhs});             \
  }                                                                           \
  TensorDim operator _op_(const TensorDim& lhs, size_t rhs) {                 \
    return TensorFriend::DimOp(_poly_op_, {lhs, TensorDim{rhs}});             \
  }

TILE_CC_DEFINE_TENSOR_IDXDIM_BINARY_OPS(+, TILE_POLY_OP_ADD);
TILE_CC_DEFINE_TENSOR_IDXDIM_BINARY_OPS(-, TILE_POLY_OP_SUB);
TILE_CC_DEFINE_TENSOR_IDXDIM_BINARY_OPS(*, TILE_POLY_OP_MUL);
TILE_CC_DEFINE_TENSOR_IDXDIM_BINARY_OPS(/, TILE_POLY_OP_DIV);

Constraint TensorIndex::operator<(size_t rhs) const {
  ffi_call_void(tile_poly_expr_add_constraint, impl_->ptr.get(), rhs);
  return Constraint();
}

Constraint TensorIndex::operator<(const TensorDim& rhs) const {
  if (!rhs.impl_->size) {
    throw std::runtime_error("Undefined dimension.");
  }
  ffi_call_void(tile_poly_expr_add_constraint, impl_->ptr.get(), *rhs.impl_->size);
  return Constraint();
}

std::string TensorIndex::str() const { return ffi_str(ffi_call<tile_string*>(tile_poly_expr_repr, impl_->ptr.get())); }

TensorDim::TensorDim() : impl_(std::make_shared<Impl>()) {}

TensorDim::TensorDim(size_t value) : impl_(std::make_shared<Impl>()) { impl_->size.reset(new size_t(value)); }

Tensor::Tensor(const TensorShape& shape) : impl_(new Impl) {
  impl_->ptr = make_tile_expr(     //
      ffi_call<tile_expr*>(        //
          tile_expr_param,         //
          shape.impl_->ptr.get(),  //
          ""));
}

Tensor::Tensor(const std::string& name, const TensorShape& shape) : impl_(new Impl) {
  impl_->ptr = make_tile_expr(     //
      ffi_call<tile_expr*>(        //
          tile_expr_param,         //
          shape.impl_->ptr.get(),  //
          name.c_str()));
}

Tensor::Tensor(const std::initializer_list<TensorDim>& dims) : impl_(new Impl) { impl_->dims = dims; }

Tensor::Tensor(const std::vector<TensorDim>& dims) : impl_(new Impl) { impl_->dims = dims; }

Tensor::Tensor(const std::string& name, const std::vector<TensorDim>& dims) : impl_(new Impl) {
  impl_->dims = dims;
  impl_->name = name;
}

Tensor::Tensor(const std::string& name, const std::initializer_list<TensorDim>& dims) : impl_(new Impl) {
  impl_->dims = dims;
  impl_->name = name;
}

Tensor::Tensor(int value) : impl_(new Impl) {  //
  impl_->ptr = make_tile_expr(ffi_call<tile_expr*>(tile_expr_int, value));
}

Tensor::Tensor(int64_t value) : impl_(new Impl) {
  impl_->ptr = make_tile_expr(ffi_call<tile_expr*>(tile_expr_int, value));
}

Tensor::Tensor(double value) : impl_(new Impl) {
  impl_->ptr = make_tile_expr(ffi_call<tile_expr*>(tile_expr_float, value));
}

Tensor::~Tensor() = default;

// Impl Constructor
Tensor::Tensor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

// Copy Constructor
Tensor::Tensor(const Tensor& rhs) { *this = rhs; }

// Copy Assignment
Tensor& Tensor::operator=(const Tensor& rhs) {
  if (this != &rhs) {
    impl_.reset(new Impl(*rhs.impl_));
  }
  return *this;
}

std::string Tensor::str() const { return ffi_str(ffi_call<tile_string*>(tile_expr_repr, impl_->ptr.get())); }

void Tensor::bind_dims(const std::vector<TensorDim>& dims) const {
  std::vector<size_t> sizes;
  if (impl_->dims.size()) {
    // this handles intermediate temporaries (results of previous outputs)
    for (const auto& dim : impl_->dims) {
      if (!dim.impl_->size) {
        throw std::runtime_error("Undefined dimension.");
      }
      sizes.emplace_back(*dim.impl_->size);
    }
  } else {
    // this is the fallback which handles any other case
    auto this_shape = shape();
    for (size_t i = 0; i < this_shape.rank(); i++) {
      sizes.emplace_back(this_shape.size_at(i));
    }
  }

  if (dims.size() != sizes.size()) {
    std::stringstream ss;
    ss << "TensorShape mismatch in bind_dims(). "
       << "Tensor shape: " << sizes.size() << ", dims: " << dims.size();
    throw std::runtime_error(ss.str());
  }

  for (size_t i = 0; i < dims.size(); i++) {
    auto& dim = dims[i];
    if (!dim.impl_->size) {
      dim.impl_->size.reset(new size_t(sizes[i]));
    } else if (sizes[i] != *dim.impl_->size) {
      std::stringstream ss;
      ss << "bind_dims() mismatch on dim " << i << ". "
         << "Required: " << *dim.impl_->size << ", Actual: " << sizes[i];
      throw std::runtime_error(ss.str());
    }
  }
}

IndexedTensor Tensor::operator()(const std::vector<TensorIndex>& idxs) const {
  std::vector<size_t> sizes;
  for (const auto& dim : impl_->dims) {
    if (!dim.impl_->size) {
      throw std::runtime_error("Undefined dimension.");
    }
    sizes.emplace_back(*dim.impl_->size);
  }
  std::vector<tile_poly_expr*> idx_ptrs(idxs.size());
  for (size_t i = 0; i < idxs.size(); i++) {
    idx_ptrs[i] = TensorFriend::GetImpl(idxs[i])->ptr.get();
  }
  auto impl = std::make_unique<IndexedTensor::Impl>();
  impl->src = impl_.get();
  impl->unary = make_tile_expr(   //
      ffi_call<tile_expr*>(       //
          tile_expr_tensor_spec,  //
          impl_->ptr.get(),       //
          idx_ptrs.size(),        //
          idx_ptrs.data(),        //
          sizes.data()));
  return IndexedTensor{std::move(impl)};
}

size_t Tensor::dims(const size_t dim) const {
  auto this_shape = shape();
  if (this_shape.rank() <= dim) {
    throw std::runtime_error("Requested dimension number higher than number of tensor dimensions");
  }
  return this_shape.size_at(dim);
}

Tensor Tensor::operator-() const { return Call("neg", {*this}); }
Tensor Tensor::operator~() const { return Call("bit_not", {*this}); }

#define TILE_CC_DEFINE_TENSOR_BINARY_OPS(_op_, _fn_)                                            \
  Tensor operator _op_(const Tensor& lhs, const Tensor& rhs) { return Call(_fn_, lhs, rhs); }   \
  Tensor operator _op_(const Tensor& lhs, int rhs) { return Call(_fn_, lhs, Tensor{rhs}); }     \
  Tensor operator _op_(const Tensor& lhs, int64_t rhs) { return Call(_fn_, lhs, Tensor{rhs}); } \
  Tensor operator _op_(const Tensor& lhs, double rhs) { return Call(_fn_, lhs, Tensor{rhs}); }  \
  Tensor operator _op_(int lhs, const Tensor& rhs) { return Call(_fn_, Tensor{lhs}, rhs); }     \
  Tensor operator _op_(int64_t lhs, const Tensor& rhs) { return Call(_fn_, Tensor{lhs}, rhs); } \
  Tensor operator _op_(double lhs, const Tensor& rhs) { return Call(_fn_, Tensor{lhs}, rhs); }

TILE_CC_DEFINE_TENSOR_BINARY_OPS(+, "add");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(-, "sub");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(*, "mul");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(/, "div");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(==, "cmp_eq");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(!=, "cmp_ne");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(<, "cmp_lt");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(>, "cmp_gt");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(<=, "cmp_le");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(>=, "cmp_ge");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(<<, "bit_left");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(>>, "bit_right");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(&, "bit_and");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(|, "bit_or");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(^, "bit_xor");

Tensor& Tensor::no_defract() {
  ffi_call_void(tile_expr_contraction_set_no_defract, impl_->ptr.get(), true);
  return *this;
}

Tensor& Tensor::use_default(const Tensor& rhs) {
  ffi_call_void(tile_expr_contraction_set_use_default, impl_->ptr.get(), rhs.impl_->ptr.get());
  return *this;
}

TensorShape Tensor::shape() const {
  auto impl = std::make_shared<TensorShape::Impl>();
  impl->ptr = make_tile_shape(       //
      ffi_call<tile_shape*>(         //
          tile_expr_evaluate_shape,  //
          impl_->ptr.get()));
  return TensorShape(impl);
}

IndexedTensor::~IndexedTensor() = default;

IndexedTensor::IndexedTensor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

IndexedTensor::IndexedTensor(IndexedTensor&& rhs) noexcept : impl_(std::move(rhs.impl_)) {}

void IndexedTensor::Impl::MakeContraction(tile_agg_op agg_op, const IndexedTensor& rhs) {
  tile_combo_op combo_op;
  std::vector<tile_expr*> inputs;
  if (rhs.impl_->unary) {
    // unary op: TensorSpec
    combo_op = TILE_COMBO_OP_NONE;
    inputs.emplace_back(rhs.impl_->unary.get());
  } else if (rhs.impl_->nary) {
    // binary/ternary op: ComboParts
    combo_op = rhs.impl_->nary->op;
    inputs = rhs.impl_->nary->args;
  } else {
    throw std::runtime_error("Invalid impl");
  }
  src->ptr = make_tile_expr(      //
      ffi_call<tile_expr*>(       //
          tile_expr_contraction,  //
          agg_op,                 //
          combo_op,               //
          unary.get(),            //
          inputs.size(),          //
          inputs.data(),          //
          src->name.c_str()));
}

IndexedTensor& IndexedTensor::operator+=(const IndexedTensor& rhs) {
  impl_->MakeContraction(TILE_AGG_OP_SUM, rhs);
  return *this;
}

IndexedTensor& IndexedTensor::operator*=(const IndexedTensor& rhs) {
  impl_->MakeContraction(TILE_AGG_OP_PROD, rhs);
  return *this;
}

IndexedTensor& IndexedTensor::operator>=(const IndexedTensor& rhs) {
  impl_->MakeContraction(TILE_AGG_OP_MAX, rhs);
  return *this;
}

IndexedTensor& IndexedTensor::operator<=(const IndexedTensor& rhs) {
  impl_->MakeContraction(TILE_AGG_OP_MIN, rhs);
  return *this;
}

IndexedTensor& IndexedTensor::operator=(const IndexedTensor& rhs) {
  impl_->MakeContraction(TILE_AGG_OP_ASSIGN, rhs);
  return *this;
}

IndexedTensor IndexedTensor::operator+(const IndexedTensor& rhs) const {
  return TensorFriend::ComboParts(TILE_COMBO_OP_ADD, {this, &rhs});
}

IndexedTensor IndexedTensor::operator*(const IndexedTensor& rhs) const {
  return TensorFriend::ComboParts(TILE_COMBO_OP_MUL, {this, &rhs});
}

IndexedTensor IndexedTensor::operator==(const IndexedTensor& rhs) const {
  return TensorFriend::ComboParts(TILE_COMBO_OP_EQ, {this, &rhs});
}

IndexedTensor cond(const IndexedTensor& lhs, const IndexedTensor& rhs, const IndexedTensor& true_case) {
  return TensorFriend::ComboParts(TILE_COMBO_OP_COND, {&lhs, &rhs, &true_case});
}

Tensor Call(const std::string& fn, const std::vector<Tensor>& args) { return TensorFriend::Call(fn, args); }

}  // namespace edsl
}  // namespace plaidml
}  // namespace vertexai
