// Copyright 2019 Intel Corporation.

#include "plaidml/edsl/ffi.h"

#include <sstream>

#include <boost/format.hpp>

#include "base/util/logging.h"
#include "plaidml/edsl/internal.h"
#include "tile/lang/ast.h"

using namespace vertexai::tile;        // NOLINT
using namespace vertexai::tile::lang;  // NOLINT

namespace {

static std::atomic<size_t> next_idx_id{0};

template <typename T, typename F>
T ffi_wrap(tile_error* err, T val, F fn) {
  try {
    err->code = 0;
    err->msg = nullptr;
    return fn();
  } catch (const std::exception& ex) {
    err->code = 1;
    err->msg = new tile_string{ex.what()};
    return val;
  } catch (...) {
    err->code = 1;
    err->msg = new tile_string{"C++ exception"};
    return val;
  }
}

template <typename F>
void ffi_wrap_void(tile_error* err, F fn) {
  try {
    err->code = 0;
    err->msg = nullptr;
    fn();
  } catch (const std::exception& ex) {
    err->code = 1;
    err->msg = new tile_string{ex.what()};
  } catch (...) {
    err->code = 1;
    err->msg = new tile_string{"C++ exception"};
  }
}

AggregationOp into_agg_op(tile_agg_op op) {
  switch (op) {
    case TILE_AGG_OP_NONE:
      return AggregationOp::NONE;
    case TILE_AGG_OP_SUM:
      return AggregationOp::SUM;
    case TILE_AGG_OP_PROD:
      return AggregationOp::PROD;
    case TILE_AGG_OP_MIN:
      return AggregationOp::MIN;
    case TILE_AGG_OP_MAX:
      return AggregationOp::MAX;
    case TILE_AGG_OP_ASSIGN:
      return AggregationOp::ASSIGN;
  }
  throw std::runtime_error("Invalid agg_op");
}

CombinationOp into_combo_op(tile_combo_op op) {
  switch (op) {
    case TILE_COMBO_OP_NONE:
      return CombinationOp::NONE;
    case TILE_COMBO_OP_ADD:
      return CombinationOp::PLUS;
    case TILE_COMBO_OP_MUL:
      return CombinationOp::MULTIPLY;
    case TILE_COMBO_OP_COND:
      return CombinationOp::COND;
    case TILE_COMBO_OP_EQ:
      return CombinationOp::EQ;
  }
  throw std::runtime_error("Invalid combo_op");
}

}  // namespace

extern "C" {

const char* tile_string_ptr(tile_string* str) { return str->str.c_str(); }

void tile_string_free(tile_string* str) {
  tile_error err;
  ffi_wrap_void(&err, [&] {  //
    delete str;
  });
}

tile_shape* tile_shape_alloc(tile_error* err, plaidml_datatype dtype, const char* layout) {
  return ffi_wrap<tile_shape*>(err, nullptr, [&] {
    std::vector<TensorDimension> dims;
    return new tile_shape{TensorShape(static_cast<DataType>(dtype), dims, layout)};
  });
}

tile_string* tile_shape_repr(tile_error* err, tile_shape* shape) {
  return ffi_wrap<tile_string*>(err, nullptr, [&] {
    std::stringstream ss;
    ss << shape->shape;
    return new tile_string{ss.str()};
  });
}

void tile_shape_add_dimension(tile_error* err, tile_shape* shape, uint64_t size, int64_t stride) {
  ffi_wrap_void(err, [&] {  //
    shape->shape.dims.emplace_back(stride, size);
  });
}

size_t tile_shape_get_rank(tile_error* err, tile_shape* shape) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return shape->shape.dims.size();
  });
}

plaidml_datatype tile_shape_get_type(tile_error* err, tile_shape* shape) {
  return ffi_wrap<plaidml_datatype>(err, PLAIDML_DATA_INVALID, [&] {  //
    return static_cast<plaidml_datatype>(shape->shape.type);
  });
}

uint64_t tile_shape_get_dimension_size(tile_error* err, tile_shape* shape, size_t dim) {
  return ffi_wrap<uint64_t>(err, 0, [&] {  //
    return shape->shape.dims.at(dim).size;
  });
}

int64_t tile_shape_get_dimension_stride(tile_error* err, tile_shape* shape, size_t dim) {
  return ffi_wrap<int64_t>(err, 0, [&] {  //
    return shape->shape.dims.at(dim).stride;
  });
}

uint64_t tile_shape_get_byte_size(tile_error* err, tile_shape* shape) {
  return ffi_wrap<uint64_t>(err, 0, [&] {  //
    return shape->shape.byte_size();
  });
}

const void* tile_shape_get_ptr(tile_error* err, tile_shape* shape) {
  return ffi_wrap<const void*>(err, 0, [&] {  //
    return &shape->shape;
  });
}

void tile_shape_free(tile_error* err, tile_shape* shape) {
  ffi_wrap_void(err, [&] {  //
    delete shape;
  });
}

void tile_expr_free(tile_error* err, tile_expr* expr) {
  ffi_wrap_void(err, [&] {  //
    delete expr;
  });
}

tile_string* tile_expr_repr(tile_error* err, tile_expr* expr) {
  return ffi_wrap<tile_string*>(err, nullptr, [&] {  //
    return new tile_string{expr->expr->str()};
  });
}

tile_expr* tile_expr_param(tile_error* err, tile_shape* shape, const char* name) {
  return ffi_wrap<tile_expr*>(  //
      err, nullptr,
      [&] {  //
        return new tile_expr{std::make_shared<ParamExpr>(shape->shape, name)};
      });
}

tile_expr* tile_expr_int(tile_error* err, int64_t value) {
  return ffi_wrap<tile_expr*>(err, nullptr, [&] {  //
    return new tile_expr{std::make_shared<IntConst>(value)};
  });
}

int64_t tile_expr_int_get_value(tile_error* err, tile_expr* expr) {
  return ffi_wrap<int64_t>(err, 0, [&] {
    if (!expr) {
      throw std::runtime_error("tile_expr_int_get_value can only be used on an IntConst");
    }
    auto int_expr = std::dynamic_pointer_cast<IntConst>(expr->expr);
    if (!int_expr) {
      throw std::runtime_error("tile_expr_int_get_value can only be used on an IntConst");
    }
    return int_expr->value;
  });
}

tile_expr* tile_expr_float(tile_error* err, double value) {
  return ffi_wrap<tile_expr*>(err, nullptr, [&] {  //
    return new tile_expr{std::make_shared<FloatConst>(value)};
  });
}

double tile_expr_float_get_value(tile_error* err, tile_expr* expr) {
  return ffi_wrap<double>(err, 0, [&] {
    auto float_expr = std::dynamic_pointer_cast<FloatConst>(expr->expr);
    if (!float_expr) {
      throw std::runtime_error("tile_expr_float_get_value can only be used on an FloatConst");
    }
    return float_expr->value;
  });
}

tile_expr* tile_expr_call(tile_error* err, const char* fn, size_t nargs, tile_expr** args) {
  return ffi_wrap<tile_expr*>(err, nullptr, [&] {
    std::vector<std::shared_ptr<Expr>> exprs(nargs);
    for (size_t i = 0; i < nargs; i++) {
      if (!args[i]) {
        throw std::runtime_error(str(boost::format("Undefined tensor in call to %1%()") % fn));
      }
      exprs[i] = args[i]->expr;
    }
    return new tile_expr{std::make_shared<CallExpr>(fn, exprs)};
  });
}

tile_expr* tile_expr_tensor_spec(tile_error* err,              //
                                 tile_expr* ref,               //
                                 size_t rank,                  //
                                 tile_poly_expr** input_idxs,  //
                                 size_t* output_sizes) {
  return ffi_wrap<tile_expr*>(err, nullptr, [&] {
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

tile_expr* tile_expr_contraction(tile_error* err,         //
                                 tile_agg_op agg_op,      //
                                 tile_combo_op combo_op,  //
                                 tile_expr* raw_output,   //
                                 size_t ninputs,          //
                                 tile_expr** raw_inputs,  //
                                 const char* name) {
  return ffi_wrap<tile_expr*>(err, nullptr, [&] {
    auto output = std::dynamic_pointer_cast<TensorSpecExpr>(raw_output->expr);
    if (!output) {
      throw std::runtime_error("oops: out_spec");
    }
    auto expr = std::make_shared<ContractionExpr>();
    expr->name = name;
    expr->agg_op = into_agg_op(agg_op);
    expr->combo_op = into_combo_op(combo_op);
    expr->output = output;
    for (size_t i = 0; i < ninputs; i++) {
      auto input = std::dynamic_pointer_cast<TensorSpecExpr>(raw_inputs[i]->expr);
      if (!input) {
        throw std::runtime_error("oops: input");
      }
      expr->inputs.emplace_back(input);
    }
    ConstraintCollector cc;
    for (const auto& idx : output->index_spec) {
      idx->Accept(&cc);
    }
    for (const auto& tensor : expr->inputs) {
      for (const auto& idx : tensor->index_spec) {
        idx->Accept(&cc);
      }
    }
    expr->constraints = cc.constraints;
    return new tile_expr{expr};
  });
}

void tile_expr_contraction_set_no_defract(tile_error* err, tile_expr* expr, bool no_defract) {
  ffi_wrap_void(err, [&] {
    auto cion = std::dynamic_pointer_cast<ContractionExpr>(expr->expr);
    if (!cion) {
      throw std::runtime_error("no_defract can only be specified on a contraction.");
    }
    cion->no_defract = no_defract;
  });
}

void tile_expr_contraction_set_use_default(tile_error* err, tile_expr* expr, tile_expr* use_default) {
  ffi_wrap_void(err, [&] {
    auto cion = std::dynamic_pointer_cast<ContractionExpr>(expr->expr);
    if (!cion) {
      throw std::runtime_error("use_default can only be specified on a contraction.");
    }
    cion->use_default = use_default->expr;
  });
}

tile_shape* tile_expr_evaluate_shape(tile_error* err, tile_expr* expr) {
  return ffi_wrap<tile_shape*>(err, nullptr, [&] {  //
    return new tile_shape{EvaluateShape(expr->expr)};
  });
}

void tile_poly_expr_free(tile_error* err, tile_poly_expr* expr) {
  ffi_wrap_void(err, [&] {  //
    delete expr;
  });
}

tile_string* tile_poly_expr_repr(tile_error* err, tile_poly_expr* expr) {
  return ffi_wrap<tile_string*>(err, nullptr, [&] {  //
    return new tile_string{expr->expr->str()};
  });
}

tile_poly_expr* tile_poly_expr_index(tile_error* err, const char* name) {
  return ffi_wrap<tile_poly_expr*>(err, nullptr, [&] {  //
    return new tile_poly_expr{std::make_shared<PolyIndex>(next_idx_id++, std::string{name})};
  });
}

tile_poly_expr* tile_poly_expr_literal(tile_error* err, int64_t value) {
  return ffi_wrap<tile_poly_expr*>(err, nullptr, [&] {  //
    return new tile_poly_expr{std::make_shared<PolyLiteral>(value)};
  });
}

tile_poly_expr* tile_poly_expr_op(tile_error* err, tile_poly_op op, size_t nargs, tile_poly_expr** args) {
  return ffi_wrap<tile_poly_expr*>(err, nullptr, [&] {
    std::vector<std::shared_ptr<PolyExpr>> vec_args(nargs);
    for (size_t i = 0; i < nargs; i++) {
      vec_args[i] = args[i]->expr;
    }
    return new tile_poly_expr{std::make_shared<PolyOpExpr>(static_cast<PolyOp>(op), vec_args)};
  });
}

void tile_poly_expr_add_constraint(tile_error* err, tile_poly_expr* lhs, size_t rhs) {
  ffi_wrap_void(err, [&] {
    auto constraint = std::make_shared<ConstraintExpr>(lhs->expr, rhs);
    ConstraintApplier applier(constraint);
    lhs->expr->Accept(&applier);
  });
}

void tile_program_free(tile_error* err, tile_program* program) {
  ffi_wrap_void(err, [&] {  //
    delete program;
  });
}

tile_program* tile_program_evaluate(tile_error* err, const char* name, size_t nexprs, tile_expr** raw_exprs) {
  return ffi_wrap<tile_program*>(err, nullptr, [&] {
    std::vector<std::shared_ptr<Expr>> exprs(nexprs);
    for (size_t i = 0; i < nexprs; i++) {
      if (!raw_exprs[i]) {
        throw std::runtime_error("Undefined expression in tile_program_evaluate");
      }
      exprs[i] = raw_exprs[i]->expr;
    }
    return new tile_program{Evaluate(name, exprs)};
  });
}

tile_string* tile_program_repr(tile_error* err, tile_program* program) {
  return ffi_wrap<tile_string*>(err, nullptr, [&] {  //
    return new tile_string{to_string(program->eval.runinfo.program)};
  });
}

const void* tile_program_runinfo(tile_error* err, tile_program* program) {
  return ffi_wrap<const void*>(err, nullptr, [&] {  //
    return &program->eval.runinfo;
  });
}

}  // extern "C"
