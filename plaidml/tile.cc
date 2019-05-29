// Copyright 2019 Intel Corporation.

#include <sstream>

#include "base/util/logging.h"
#include "plaidml/plaidml.h"
#include "tile/lang/ast.h"

using namespace vertexai::tile;        // NOLINT
using namespace vertexai::tile::lang;  // NOLINT

extern "C" {

typedef enum {
  TILE_POLY_OP_NEG,
  TILE_POLY_OP_ADD,
  TILE_POLY_OP_SUB,
  TILE_POLY_OP_MUL,
  TILE_POLY_OP_DIV,
} tile_poly_op;

typedef enum {
  TILE_AGG_OP_NONE,
  TILE_AGG_OP_SUM,
  TILE_AGG_OP_MAX,
  TILE_AGG_OP_MIN,
  TILE_AGG_OP_PROD,
  TILE_AGG_OP_ASSIGN
} tile_agg_op;

typedef enum {
  TILE_COMBO_OP_NONE,
  TILE_COMBO_OP_MUL,
  TILE_COMBO_OP_ADD,
  TILE_COMBO_OP_EQ,
  TILE_COMBO_OP_COND,
} tile_combo_op;

struct tile_string {
  std::string str;
};

struct tile_error {
  size_t code;
  tile_string* msg;
};

struct tile_shape {
  TensorShape shape;
};

struct tile_expr {
  std::shared_ptr<Expr> expr;
};

struct tile_poly_expr {
  std::shared_ptr<PolyExpr> expr;
};

struct tile_ret_void {
  tile_error err;
};

#define TILE_DEFINE_RET(_name_, _type_) \
  struct tile_ret_##_name_ {            \
    tile_error err;                     \
    _type_ ret;                         \
  };

TILE_DEFINE_RET(string, tile_string*)
TILE_DEFINE_RET(uint, size_t)
TILE_DEFINE_RET(i64, int64_t)
TILE_DEFINE_RET(u64, uint64_t)
TILE_DEFINE_RET(shape, tile_shape*)
TILE_DEFINE_RET(expr, tile_expr*)
TILE_DEFINE_RET(poly_expr, tile_poly_expr*)

}  // extern "C"

namespace {

tile_error tile_error_ok() { return tile_error{0, nullptr}; }

template <typename T, typename V, typename F>
T ffi_wrap(V val, F fn) {
  try {
    return T{tile_error_ok(), fn()};
  } catch (const std::exception& ex) {
    return T{tile_error{1, new tile_string{ex.what()}}, val};
  } catch (...) {
    return T{tile_error{1, new tile_string{"C++ exception"}}, val};
  }
}

template <typename F>
tile_ret_void ffi_wrap_void(F fn) {
  try {
    fn();
    return tile_ret_void{tile_error_ok()};
  } catch (const std::exception& ex) {
    return tile_ret_void{tile_error{1, new tile_string{ex.what()}}};
  } catch (...) {
    return tile_ret_void{tile_error{1, new tile_string{"C++ exception"}}};
  }
}

}  // namespace

extern "C" {

const char* tile_string_ptr(tile_string* str) { return str->str.c_str(); }

tile_ret_void tile_string_free(tile_string* str) {
  return ffi_wrap_void([&] { delete str; });
}

tile_ret_shape tile_shape_alloc(plaidml_datatype dtype) {
  return ffi_wrap<tile_ret_shape>(nullptr, [&] {
    std::vector<TensorDimension> dims;
    return new tile_shape{TensorShape{static_cast<DataType>(dtype), dims}};
  });
}

tile_ret_string tile_shape_repr(tile_shape* shape) {
  return ffi_wrap<tile_ret_string>(nullptr, [&] {
    std::stringstream ss;
    ss << shape->shape;
    return new tile_string{ss.str()};
  });
}

tile_ret_void tile_shape_add_dimension(tile_shape* shape, uint64_t size, int64_t stride) {
  return ffi_wrap_void([&] { shape->shape.dims.emplace_back(stride, size); });
}

tile_ret_uint tile_shape_get_rank(tile_shape* shape) {
  return ffi_wrap<tile_ret_uint, size_t>(0, [&] { return shape->shape.dims.size(); });
}

tile_ret_u64 tile_shape_get_dimension_size(tile_shape* shape, size_t dim) {
  return ffi_wrap<tile_ret_u64, uint64_t>(0, [&] { return shape->shape.dims.at(dim).size; });
}

tile_ret_i64 tile_shape_get_dimension_stride(tile_shape* shape, size_t dim) {
  return ffi_wrap<tile_ret_i64, int64_t>(0, [&] { return shape->shape.dims.at(dim).stride; });
}

tile_ret_void tile_shape_free(tile_shape* shape) {
  return ffi_wrap_void([&] { delete shape; });
}

tile_ret_void tile_expr_free(tile_expr* expr) {
  return ffi_wrap_void([&] { delete expr; });
}

tile_ret_string tile_expr_repr(tile_expr* expr) {
  return ffi_wrap<tile_ret_string>(nullptr, [&] { return new tile_string{expr->expr->str()}; });
}

tile_ret_expr tile_expr_param(tile_shape* shape, const char* name) {
  return ffi_wrap<tile_ret_expr>(nullptr,
                                 [&] { return new tile_expr{std::make_shared<ParamExpr>(shape->shape, name)}; });
}

tile_ret_expr tile_expr_int(int64_t value) {
  return ffi_wrap<tile_ret_expr>(nullptr, [&] { return new tile_expr{std::make_shared<IntConst>(value)}; });
}

tile_ret_expr tile_expr_float(double value) {
  return ffi_wrap<tile_ret_expr>(nullptr, [&] { return new tile_expr{std::make_shared<FloatConst>(value)}; });
}

tile_ret_expr tile_expr_call(const char* fn, size_t nargs, tile_expr** args) {
  return ffi_wrap<tile_ret_expr>(nullptr, [&] {
    std::vector<std::shared_ptr<Expr>> exprs(nargs);
    for (size_t i = 0; i < nargs; i++) {
      exprs[i] = args[i]->expr;
    }
    return new tile_expr{std::make_shared<CallExpr>(fn, exprs)};
  });
}

tile_ret_expr tile_expr_tensor_spec(tile_expr* ref, size_t rank, tile_poly_expr** input_idxs, size_t* output_sizes) {
  return ffi_wrap<tile_ret_expr>(nullptr, [&] {
    std::vector<size_t> vec_sizes;
    std::vector<std::shared_ptr<PolyExpr>> vec_idxs(rank);
    for (size_t i = 0; i < rank; i++) {
      vec_idxs[i] = input_idxs[i]->expr;
      if (output_sizes) {
        vec_sizes.emplace_back(output_sizes[i]);
      }
    }
    return new tile_expr{std::make_shared<TensorSpecExpr>(ref ? ref->expr : nullptr, vec_idxs, vec_sizes)};
  });
}

tile_ret_expr tile_expr_contraction(tile_agg_op agg_op,           //
                                    tile_combo_op combo_op,       //
                                    tile_expr* raw_output,        //
                                    size_t ninputs,               //
                                    tile_expr** raw_inputs,       //
                                    size_t nconstraints,          //
                                    tile_expr** raw_constraints,  //
                                    bool no_defract,              //
                                    tile_expr* raw_use_default) {
  return ffi_wrap<tile_ret_expr>(nullptr, [&] {
    // std::vector<std::shared_ptr<ConstraintExpr>> constraints;
    auto output = std::dynamic_pointer_cast<TensorSpecExpr>(raw_output->expr);
    if (!output) {
      throw std::runtime_error("oops: out_spec");
    }
    auto expr = std::make_shared<ContractionExpr>();
    expr->agg_op = AggregationOp::SUM;
    expr->combo_op = CombinationOp::MULTIPLY;
    expr->output = output;
    for (size_t i = 0; i < ninputs; i++) {
      auto input = std::dynamic_pointer_cast<TensorSpecExpr>(raw_inputs[i]->expr);
      if (!input) {
        throw std::runtime_error("oops: input");
      }
      expr->inputs.emplace_back(input);
    }
    for (size_t i = 0; i < nconstraints; i++) {
    }
    expr->no_defract = no_defract;
    if (raw_use_default) {
      expr->use_default = raw_use_default->expr;
    }
    return new tile_expr{expr};
  });
}

tile_ret_shape tile_expr_evaluate_shape(tile_expr* expr) {
  return ffi_wrap<tile_ret_shape>(nullptr, [&] { return new tile_shape{EvaluateShape(expr->expr)}; });
}

tile_ret_void tile_poly_expr_free(tile_poly_expr* expr) {
  return ffi_wrap_void([&] { delete expr; });
}

tile_ret_string tile_poly_expr_repr(tile_poly_expr* expr) {
  return ffi_wrap<tile_ret_string>(nullptr, [&] { return new tile_string{expr->expr->str()}; });
}

tile_ret_poly_expr tile_poly_expr_index(const void* ptr, const char* name) {
  return ffi_wrap<tile_ret_poly_expr>(
      nullptr, [&] { return new tile_poly_expr{std::make_shared<PolyIndex>(ptr, std::string{name})}; });
}

tile_ret_poly_expr tile_poly_expr_literal(int64_t value) {
  return ffi_wrap<tile_ret_poly_expr>(nullptr,
                                      [&] { return new tile_poly_expr{std::make_shared<PolyLiteral>(value)}; });
}

tile_ret_poly_expr tile_poly_expr_op(tile_poly_op op, size_t nargs, tile_poly_expr** args) {
  return ffi_wrap<tile_ret_poly_expr>(nullptr, [&] {
    std::vector<std::shared_ptr<PolyExpr>> vec_args(nargs);
    for (size_t i = 0; i < nargs; i++) {
      vec_args[i] = args[i]->expr;
    }
    return new tile_poly_expr{std::make_shared<PolyOpExpr>(static_cast<PolyOp>(op), vec_args)};
  });
}

}  // extern "C"
