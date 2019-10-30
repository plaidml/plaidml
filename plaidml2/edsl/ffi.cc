// Copyright 2019 Intel Corporation.

#include "plaidml2/edsl/ffi.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <boost/format.hpp>

#include "base/util/env.h"
#include "base/util/logging.h"
#include "plaidml2/core/internal.h"
#include "plaidml2/edsl/derivs.h"
#include "pmlc/util/enums.h"
#include "tile/lang/ast/ast.h"
#include "tile/lang/ast/gradient.h"

using namespace vertexai::tile;             // NOLINT
using namespace vertexai::tile::lang;       // NOLINT
using namespace vertexai::tile::lang::ast;  // NOLINT

using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using plaidml::core::GlobalContext;
using pmlc::util::AggregationKind;
using pmlc::util::CombinationKind;

namespace {

static std::atomic<size_t> next_idx_id{0};

AggregationOp into_agg_op(plaidml_agg_op op) {
  switch (op) {
    case PLAIDML_AGG_OP_NONE:
      return AggregationOp::NONE;
    case PLAIDML_AGG_OP_SUM:
      return AggregationOp::SUM;
    case PLAIDML_AGG_OP_PROD:
      return AggregationOp::PROD;
    case PLAIDML_AGG_OP_MIN:
      return AggregationOp::MIN;
    case PLAIDML_AGG_OP_MAX:
      return AggregationOp::MAX;
    case PLAIDML_AGG_OP_ASSIGN:
      return AggregationOp::ASSIGN;
  }
  throw std::runtime_error("Invalid agg_op");
}

CombinationOp into_combo_op(plaidml_combo_op op) {
  switch (op) {
    case PLAIDML_COMBO_OP_NONE:
      return CombinationOp::NONE;
    case PLAIDML_COMBO_OP_ADD:
      return CombinationOp::PLUS;
    case PLAIDML_COMBO_OP_MUL:
      return CombinationOp::MULTIPLY;
    case PLAIDML_COMBO_OP_COND:
      return CombinationOp::COND;
    case PLAIDML_COMBO_OP_EQ:
      return CombinationOp::EQ;
  }
  throw std::runtime_error("Invalid combo_op");
}

AggregationKind getAggregationKind(plaidml_agg_op agg_op) {
  switch (agg_op) {
    case PLAIDML_AGG_OP_ASSIGN:
      return AggregationKind::assign;
    case PLAIDML_AGG_OP_MAX:
      return AggregationKind::max;
    case PLAIDML_AGG_OP_MIN:
      return AggregationKind::min;
    case PLAIDML_AGG_OP_PROD:
      return AggregationKind::mul;
    case PLAIDML_AGG_OP_SUM:
      return AggregationKind::add;
    default:
      break;
  }
  throw std::runtime_error("Unsupported agg_op");
}

CombinationKind getCombinationKind(plaidml_combo_op combo_op) {
  switch (combo_op) {
    case PLAIDML_COMBO_OP_ADD:
      return CombinationKind::add;
    case PLAIDML_COMBO_OP_COND:
      return CombinationKind::cond;
    case PLAIDML_COMBO_OP_EQ:
      return CombinationKind::eq;
    case PLAIDML_COMBO_OP_MUL:
      return CombinationKind::mul;
    case PLAIDML_COMBO_OP_NONE:
      return CombinationKind::none;
    default:
      break;
  }
  throw std::runtime_error("Unsupported combo_op");
}

struct GlobalContext {
  static TileBuilder* get() {
    static thread_local TileBuilder builder;
    return &builder;
  }
};

mlir::Value* MakeAffineOp(plaidml_int_op op, const std::vector<mlir::Value*> operands) {
  auto builder = GlobalContext::get();
  switch (op) {
    case PLAIDML_INT_OP_ADD:
      return builder->MakeAffineAddOp(operands);
    case PLAIDML_INT_OP_DIV:
      return builder->MakeAffineDivOp(operands);
    case PLAIDML_INT_OP_MUL:
      return builder->MakeAffineMulOp(operands);
    case PLAIDML_INT_OP_NEG:
      return builder->MakeAffineNegOp(operands);
    case PLAIDML_INT_OP_SUB:
      return builder->MakeAffineSubOp(operands);
    case PLAIDML_INT_OP_MAX:
      return builder->MakeAffineMaxOp(operands);
    case PLAIDML_INT_OP_MIN:
      return builder->MakeAffineMinOp(operands);
  }
  throw std::runtime_error("Unknown affine op");
}

static bool use_mlir() {  //
  return vertexai::env::Get("PLAIDML_MLIR") == "1";
}

}  // namespace

extern "C" {

struct plaidml_logical_shape {
  LogicalShape shape;
  DataType dtype;
  std::vector<mlir::Value*> dims;
};

struct plaidml_dim_expr {
  DimExprPtr expr;
  mlir::Value* value = nullptr;
};

struct plaidml_poly_expr {
  PolyExprPtr expr;
  mlir::Value* value = nullptr;
};

void plaidml_edsl_init(  //
    plaidml_error* err) {
  static std::once_flag is_initialized;
  ffi_wrap_void(err, [&] {
    std::call_once(is_initialized, []() {
      IVLOG(1, "plaidml_edsl_init");
      plaidml::edsl::deriv::RegisterDerivs();
    });
  });
}

plaidml_logical_shape* plaidml_logical_shape_alloc(  //
    plaidml_error* err,                              //
    plaidml_datatype dtype,                          //
    size_t ndims,                                    //
    const int64_t* dims,                             //
    const char* layout) {
  return ffi_wrap<plaidml_logical_shape*>(err, nullptr, [&] {
    auto ret = new plaidml_logical_shape;
    ret->shape.dtype = static_cast<DataType>(dtype);
    ret->shape.layout = layout;
    for (size_t i = 0; i < ndims; i++) {
      auto int_expr = std::make_shared<DimIntExpr>(dims[i]);
      ret->shape.dims.emplace_back(LogicalDim{int_expr});
    }
    return ret;
  });
}

plaidml_string* plaidml_logical_shape_repr(  //
    plaidml_error* err,                      //
    plaidml_logical_shape* shape) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    std::stringstream ss;
    ss << shape->shape.str();
    return new plaidml_string{ss.str()};
  });
}

plaidml_string* plaidml_logical_shape_get_layout(  //
    plaidml_error* err,                            //
    plaidml_logical_shape* shape) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {  //
    return new plaidml_string{shape->shape.layout};
  });
}

size_t plaidml_logical_shape_get_ndims(  //
    plaidml_error* err,                  //
    plaidml_logical_shape* shape) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return shape->shape.dims.size();
  });
}

plaidml_datatype plaidml_logical_shape_get_dtype(  //
    plaidml_error* err,                            //
    plaidml_logical_shape* shape) {
  return ffi_wrap<plaidml_datatype>(err, PLAIDML_DATA_INVALID, [&] {  //
    return static_cast<plaidml_datatype>(shape->shape.dtype);
  });
}

int64_t plaidml_logical_shape_get_dim_int(  //
    plaidml_error* err,                     //
    plaidml_logical_shape* shape,           //
    size_t dim) {
  return ffi_wrap<int64_t>(err, 0, [&] {
    auto dim_expr = shape->shape.dims.at(dim).expr;
    auto int_expr = std::dynamic_pointer_cast<DimIntExpr>(dim_expr);
    if (int_expr) {
      return int_expr->value;
    }
    return static_cast<int64_t>(0);
  });
}

plaidml_dim_expr* plaidml_logical_shape_get_dim_expr(  //
    plaidml_error* err,                                //
    plaidml_logical_shape* shape,                      //
    size_t dim) {
  return ffi_wrap<plaidml_dim_expr*>(err, 0, [&] {  //
    return new plaidml_dim_expr{shape->shape.dims.at(dim).expr};
  });
}

void plaidml_logical_shape_free(  //
    plaidml_error* err,           //
    plaidml_logical_shape* shape) {
  ffi_wrap_void(err, [&] {  //
    delete shape;
  });
}

plaidml_shape* plaidml_logical_shape_into_tensor_shape(  //
    plaidml_error* err,                                  //
    plaidml_logical_shape* shape) {
  return ffi_wrap<plaidml_shape*>(err, 0, [&] {  //
    return new plaidml_shape{IntoTensorShape(shape->shape)};
  });
}

void plaidml_expr_free(  //
    plaidml_error* err,  //
    plaidml_expr* expr) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_free> " << expr->expr->str());
    if (expr->value) {
      GlobalContext::get()->Destroy(expr->value);
    }
    delete expr;
  });
}

plaidml_logical_shape* plaidml_expr_get_shape(  //
    plaidml_error* err,                         //
    plaidml_expr* expr) {
  // TODO(MLIR)
  return ffi_wrap<plaidml_logical_shape*>(err, nullptr, [&] {  //
    IVLOG(3, "plaidml_expr_get_shape");
    if (!expr) {
      throw std::runtime_error(
          "Cannot compute shape of null expr. Perhaps you requested the shape of an unassigned tensor?");
    }
    return new plaidml_logical_shape{expr->expr->shape};
  });
}

void plaidml_expr_bind_shape(  //
    plaidml_error* err,        //
    plaidml_expr* expr,        //
    plaidml_logical_shape* shape) {
  // TODO(MLIR)
  return ffi_wrap_void(err, [&] {  //
    IVLOG(3, "plaidml_expr_bind_shape");
    auto param_expr = std::dynamic_pointer_cast<ParamExpr>(expr->expr);
    if (!param_expr) {
      throw std::runtime_error("Shape binding is only supported on ParamExprs");
    }
    param_expr->shape = shape->shape;
  });
}

void plaidml_expr_bind_dims(  //
    plaidml_error* err,       //
    plaidml_expr* expr,       //
    size_t ndims,             //
    plaidml_dim_expr** dims) {
  return ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_bind_dims> " << expr->expr->str());
    std::vector<DimExprPtr> vec_dims(ndims);
    for (size_t i = 0; i < ndims; i++) {
      vec_dims[i] = dims[i]->expr;
    }
    expr->expr->shape.bind_dims(&vec_dims);
    for (size_t i = 0; i < ndims; i++) {
      dims[i]->expr = vec_dims[i];
      if (use_mlir()) {
        IVLOG(3, "bind_dims> i: " << i << ", from: " << expr->value << ", into: " << dims[i]->value);
        IVLOG(3, "bind_dims> i: " << i << ", from: " << expr->expr->str());
        GlobalContext::get()->BindTensorDim(i, expr->value, &dims[i]->value);
      }
    }
  });
}

plaidml_string* plaidml_expr_repr(  //
    plaidml_error* err,             //
    plaidml_expr* expr) {
  // TODO(MLIR)
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {  //
    IVLOG(3, "plaidml_expr_repr");
    return new plaidml_string{expr->expr->str()};
  });
}

plaidml_expr* plaidml_expr_dim(  //
    plaidml_error* err,          //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_dim");
    if (use_mlir()) {
      return new plaidml_expr{std::make_shared<DimExprExpr>(expr->expr), expr->value};
    } else {
      return new plaidml_expr{std::make_shared<DimExprExpr>(expr->expr)};
    }
  });
}

plaidml_expr* plaidml_expr_placeholder(  //
    plaidml_error* err,                  //
    plaidml_logical_shape* shape,        //
    plaidml_buffer* buffer,              //
    const char* name) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_placeholder");
    auto expr = std::make_shared<ParamExpr>(name);
    if (buffer) {
      expr->buffer = buffer->buffer;
    }
    expr->ComputeShape(expr, shape->shape);
    std::vector<int64_t> dims(shape->shape.dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
      auto int_expr = std::dynamic_pointer_cast<DimIntExpr>(shape->shape.dims[i].expr);
      if (int_expr) {
        dims[i] = int_expr->value;
      } else {
        dims[i] = -1;
      }
    }
    if (use_mlir()) {
      auto value = GlobalContext::get()->MakePlaceholderOp(shape->shape.dtype, dims);
      return new plaidml_expr{expr, value};
    } else {
      return new plaidml_expr{expr};
    }
  });
}

void plaidml_expr_param_reset(  //
    plaidml_error* err,         //
    plaidml_expr* expr,         //
    plaidml_buffer* buffer) {
  return ffi_wrap_void(err, [&] {
    auto param_expr = std::dynamic_pointer_cast<ParamExpr>(expr->expr);
    if (param_expr) {
      param_expr->buffer = buffer->buffer;
    } else {
      throw std::runtime_error("ParamExpr value reset requested for non-ParamExpr");
    }
  });
}

plaidml_expr* plaidml_expr_clone(  //
    plaidml_error* err,            //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_clone> " << expr->expr->str());
    // TODO(MLIR): deal with clone of expr->value
    return new plaidml_expr{expr->expr, expr->value};
  });
}

plaidml_dim_expr* plaidml_expr_get_dim(  //
    plaidml_error* err,                  //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_get_dim> " << expr->expr->str());
    auto dim_expr = std::dynamic_pointer_cast<DimExprExpr>(expr->expr);
    if (!dim_expr) {
      throw std::runtime_error("plaidml_expr_get_dim can only be used on a DimExprExpr");
    }
    // TODO(MLIR): deal with clone of expr->value
    return new plaidml_dim_expr{dim_expr->expr, expr->value};
  });
}

plaidml_expr_kind plaidml_expr_get_kind(  //
    plaidml_error* err,                   //
    plaidml_expr* expr) {
  // TODO(MLIR)
  return ffi_wrap<plaidml_expr_kind>(err, PLAIDML_EXPR_NONE, [&] {
    IVLOG(3, "plaidml_expr_get_kind");
    if (std::dynamic_pointer_cast<NoneExpr>(expr->expr)) {
      return PLAIDML_EXPR_NONE;
    }
    if (std::dynamic_pointer_cast<StringExpr>(expr->expr)) {
      return PLAIDML_EXPR_STR;
    }
    if (std::dynamic_pointer_cast<IntConst>(expr->expr)) {
      return PLAIDML_EXPR_INT;
    }
    if (std::dynamic_pointer_cast<FloatConst>(expr->expr)) {
      return PLAIDML_EXPR_FLOAT;
    }
    if (std::dynamic_pointer_cast<TupleExpr>(expr->expr)) {
      return PLAIDML_EXPR_TUPLE;
    }
    if (std::dynamic_pointer_cast<DimExprExpr>(expr->expr)) {
      return PLAIDML_EXPR_DIM;
    }
    return PLAIDML_EXPR_TENSOR;
  });
}

plaidml_expr* plaidml_expr_none(  //
    plaidml_error* err            //
) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {  //
    IVLOG(3, "plaidml_expr_none");
    if (use_mlir()) {
      auto value = GlobalContext::get()->MakeNoneOp();
      return new plaidml_expr{std::make_shared<NoneExpr>(), value};
    } else {
      return new plaidml_expr{std::make_shared<NoneExpr>()};
    }
  });
}

plaidml_expr* plaidml_expr_tuple(  //
    plaidml_error* err,            //
    size_t nargs,                  //
    plaidml_expr** args) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_tuple");
    std::vector<ExprPtr> exprs(nargs);
    std::vector<mlir::Value*> tuple(nargs);
    for (size_t i = 0; i < nargs; i++) {
      exprs[i] = args[i]->expr;
      tuple[i] = args[i]->value;
    }
    if (use_mlir()) {
      auto value = GlobalContext::get()->MakeTupleOp(tuple);
      return new plaidml_expr{std::make_shared<TupleExpr>(exprs), value};
    } else {
      return new plaidml_expr{std::make_shared<TupleExpr>(exprs)};
    }
  });
}

size_t plaidml_expr_tuple_get_count(  //
    plaidml_error* err,               //
    plaidml_expr* expr) {
  // TODO(MLIR)
  return ffi_wrap<size_t>(err, 0, [&] {
    IVLOG(3, "plaidml_expr_tuple_get_count");
    auto tuple_expr = std::dynamic_pointer_cast<TupleExpr>(expr->expr);
    if (!tuple_expr) {
      throw std::runtime_error("Expression is not a tuple");
    }
    return tuple_expr->exprs.size();
  });
}

void plaidml_expr_tuple_get_exprs(  //
    plaidml_error* err,             //
    plaidml_expr* expr,             //
    size_t nexprs,                  //
    plaidml_expr** exprs) {
  return ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_tuple_get_exprs> nexprs: " << nexprs);
    auto tuple_expr = std::dynamic_pointer_cast<TupleExpr>(expr->expr);
    if (!tuple_expr) {
      throw std::runtime_error("Expression is not a tuple");
    }
    if (use_mlir()) {
      auto elts = GlobalContext::get()->GetTupleElements(expr->value);
      for (size_t i = 0; i < std::min(nexprs, tuple_expr->exprs.size()); i++) {
        exprs[i] = new plaidml_expr{tuple_expr->exprs[i], elts[i]};
      }
    } else {
      for (size_t i = 0; i < std::min(nexprs, tuple_expr->exprs.size()); i++) {
        exprs[i] = new plaidml_expr{tuple_expr->exprs[i]};
      }
    }
  });
}

plaidml_expr* plaidml_expr_str(  //
    plaidml_error* err,          //
    const char* value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_str> " << value);
    if (use_mlir()) {
      auto str_val = GlobalContext::get()->MakeStringOp(value);
      return new plaidml_expr{std::make_shared<StringExpr>(value), str_val};
    } else {
      return new plaidml_expr{std::make_shared<StringExpr>(value)};
    }
  });
}

plaidml_string* plaidml_expr_str_get_value(  //
    plaidml_error* err,                      //
    plaidml_expr* expr) {
  // TODO(MLIR)
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_str_get_value");
    if (!expr) {
      throw std::runtime_error("plaidml_expr_str_get_value can only be used on an StringExpr");
    }
    auto str_expr = std::dynamic_pointer_cast<StringExpr>(expr->expr);
    if (!str_expr) {
      throw std::runtime_error("plaidml_expr_str_get_value can only be used on an StringExpr");
    }
    return new plaidml_string{str_expr->value};
  });
}

plaidml_expr* plaidml_expr_int(  //
    plaidml_error* err,          //
    int64_t value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_int> " << value);
    if (use_mlir()) {
      auto const_val = GlobalContext::get()->MakeScalarConstantOp(value);
      return new plaidml_expr{std::make_shared<IntConst>(value), const_val};
    } else {
      return new plaidml_expr{std::make_shared<IntConst>(value)};
    }
  });
}

int64_t plaidml_expr_int_get_value(  //
    plaidml_error* err,              //
    plaidml_expr* expr) {
  // TODO(MLIR)
  return ffi_wrap<int64_t>(err, 0, [&] {
    IVLOG(3, "plaidml_expr_int_get_value");
    if (!expr) {
      throw std::runtime_error("plaidml_expr_int_get_value can only be used on an IntConst");
    }
    auto int_expr = std::dynamic_pointer_cast<IntConst>(expr->expr);
    if (!int_expr) {
      throw std::runtime_error("plaidml_expr_int_get_value can only be used on an IntConst");
    }
    return int_expr->value;
  });
}

plaidml_expr* plaidml_expr_float(  //
    plaidml_error* err,            //
    double value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_float");
    if (use_mlir()) {
      auto const_val = GlobalContext::get()->MakeScalarConstantOp(value);
      return new plaidml_expr{std::make_shared<FloatConst>(value), const_val};
    } else {
      return new plaidml_expr{std::make_shared<FloatConst>(value)};
    }
  });
}

double plaidml_expr_float_get_value(  //
    plaidml_error* err,               //
    plaidml_expr* expr) {
  // TODO(MLIR)
  return ffi_wrap<double>(err, 0, [&] {
    IVLOG(3, "plaidml_expr_float_get_value");
    auto float_expr = std::dynamic_pointer_cast<FloatConst>(expr->expr);
    if (!float_expr) {
      throw std::runtime_error("plaidml_expr_float_get_value can only be used on an FloatConst");
    }
    return float_expr->value;
  });
}

plaidml_expr* plaidml_expr_call(  //
    plaidml_error* err,           //
    const char* fn,               //
    size_t nargs,                 //
    plaidml_expr** args) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_call");
    std::vector<ExprPtr> exprs(nargs);
    std::vector<mlir::Value*> values(nargs);
    // FIXME: has_exprs and has_values is a HACK to allow gradient callbacks
    // to work while the MLIR backend is under construction.
    bool has_exprs = true;
    bool has_values = use_mlir();
    for (size_t i = 0; i < nargs; i++) {
      if (!args[i]) {
        throw std::runtime_error(str(boost::format("Undefined tensor in call to %1%()") % fn));
      }
      if (args[i]->expr) {
        exprs[i] = args[i]->expr;
      } else {
        has_exprs = false;
      }
      if (args[i]->value) {
        values[i] = args[i]->value;
      } else {
        has_values = false;
      }
    }
    mlir::Value* value = nullptr;
    if (has_values) {
      value = GlobalContext::get()->MakePrimitiveOp(fn, values);
    }
    ExprPtr expr;
    if (has_exprs) {
      expr = MakeCall(fn, exprs);
    }
    return new plaidml_expr{expr, value};
  });
}

plaidml_expr* plaidml_expr_grad_override(  //
    plaidml_error* err,                    //
    plaidml_deriv fn,                      //
    void* user_ctx,                        //
    size_t nins,                           //
    plaidml_expr** ins,                    //
    plaidml_expr* out) {
  auto thunk = [](const ExprPtr& Y,                //
                  const ExprPtr& dY,               //
                  const std::vector<ExprPtr>& Xs,  //
                  void* user_fn,                   //
                  void* user_ctx) {
    IVLOG(6, "plaidml_grad_override> thunk");
    plaidml_deriv fn = reinterpret_cast<plaidml_deriv>(user_fn);
    std::vector<plaidml_expr*> X_exprs(Xs.size());
    std::vector<plaidml_expr*> dX_exprs(Xs.size());
    // TODO: Construct the plaidml_exprs here w/ MLIR values too?
    auto Y_expr = new plaidml_expr{Y};
    auto dY_expr = new plaidml_expr{dY};
    for (size_t i = 0; i < X_exprs.size(); i++) {
      X_exprs[i] = new plaidml_expr{Xs[i]};
    }
    fn(user_ctx, Y_expr, dY_expr, X_exprs.size(), X_exprs.data(), dX_exprs.data());
    std::vector<ExprPtr> ret(Xs.size());
    for (size_t i = 0; i < ret.size(); i++) {
      ret[i] = dX_exprs[i]->expr;
      delete dX_exprs[i];
    }
    return ret;
  };
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(5, "plaidml_grad_override");
    auto deriv_entry = std::make_shared<ExprDerivEntry>();
    deriv_entry->fn = thunk;
    deriv_entry->user_fn = reinterpret_cast<void*>(fn);
    deriv_entry->user_ctx = user_ctx;
    std::vector<ExprPtr> in_exprs(nins);
    ExprPtr out_expr;
    mlir::Value* out_value;
    for (size_t i = 0; i < nins; i++) {
      if (!ins[i]) {
        throw std::runtime_error("Undefined input tensor in gradient override");
      }
      if (ins[i]->expr) {
        in_exprs[i] = ins[i]->expr;
      } else {
        throw std::runtime_error("Currently exprs are required in plaidml_grad_override");  // TODO: MLIR
      }
    }
    if (!out) {
      throw std::runtime_error("Undefined output tensor in gradient override");
    }
    if (out->expr) {
      out_expr = out->expr;
    } else {
      throw std::runtime_error("Currently out expr is required in plaidml_expr_grad_override");  // TODO: MLIR
    }
    if (out->value && use_mlir()) {
      out_value = out->value;
    } else {
      out_value = nullptr;
    }
    ExprPtr expr = MakeGradOverride(deriv_entry, in_exprs, out_expr);
    mlir::Value* value = use_mlir() ? GlobalContext::get()->MakePrimitiveOp("ident", out_value) : nullptr;
    IVLOG(6, "The expr from plaidml_expr_grad_override has shape " << expr->shape.str());
    return new plaidml_expr{expr, value};
  });
}

plaidml_expr* plaidml_expr_index_map(  //
    plaidml_error* err,                //
    plaidml_expr* ref,                 //
    size_t ndims,                      //
    plaidml_poly_expr** raw_idxs) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_index_map");
    std::vector<std::shared_ptr<PolyExpr>> idx_exprs(ndims);
    std::vector<mlir::Value*> idx_values(ndims);
    for (size_t i = 0; i < ndims; i++) {
      idx_exprs[i] = raw_idxs[i]->expr;
      if (use_mlir()) {
        idx_values[i] = raw_idxs[i]->value;
      }
    }
    auto builder = GlobalContext::get();
    if (ref) {
      if (use_mlir()) {
        auto value = builder->MakeAffineSourceIndexMapOp(ref->value, idx_values);
        return new plaidml_expr{std::make_shared<IndexMapExpr>(ref->expr, idx_exprs), value};
      } else {
        return new plaidml_expr{std::make_shared<IndexMapExpr>(ref->expr, idx_exprs)};
      }
    }
    if (use_mlir()) {
      auto value = builder->MakeAffineSinkIndexMapOp(idx_values);
      return new plaidml_expr{std::make_shared<IndexMapExpr>(nullptr, idx_exprs), value};
    } else {
      return new plaidml_expr{std::make_shared<IndexMapExpr>(nullptr, idx_exprs)};
    }
  });
}

plaidml_expr* plaidml_expr_size_map(  //
    plaidml_error* err,               //
    size_t ndims,                     //
    plaidml_dim_expr** raw_dims) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_size_map");
    std::vector<DimExprPtr> dim_exprs;
    std::vector<mlir::Value*> dim_values;
    for (size_t i = 0; i < ndims; i++) {
      dim_exprs.emplace_back(raw_dims[i]->expr);
      if (use_mlir()) {
        dim_values.emplace_back(raw_dims[i]->value);
      }
    }
    auto builder = GlobalContext::get();
    auto value = use_mlir() ? builder->MakeAffineSizeMapOp(dim_values) : nullptr;
    return new plaidml_expr{std::make_shared<SizeMapExpr>(dim_exprs), value};
  });
}

plaidml_expr* plaidml_expr_contraction(  //
    plaidml_error* err,                  //
    plaidml_agg_op agg_op,               //
    plaidml_combo_op combo_op,           //
    plaidml_expr* raw_sink_idxs,         //
    plaidml_expr* raw_sink_sizes,        //
    size_t nsrcs,                        //
    plaidml_expr** raw_src_idxs,         //
    const char* name,                    //
    const char* layout) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_contraction");
    auto expr = std::make_shared<ContractionExpr>();
    expr->name = name;
    expr->agg_op = into_agg_op(agg_op);
    expr->combo_op = into_combo_op(combo_op);
    expr->sink_idxs = std::dynamic_pointer_cast<IndexMapExpr>(raw_sink_idxs->expr);
    if (!expr->sink_idxs) {
      throw std::runtime_error("oops: sink_idxs");
    }
    expr->sink_dims = std::dynamic_pointer_cast<SizeMapExpr>(raw_sink_sizes->expr);
    if (!expr->sink_dims) {
      throw std::runtime_error("oops: sink_dims");
    }
    std::vector<mlir::Value*> src_values;
    for (size_t i = 0; i < nsrcs; i++) {
      auto idxs = std::dynamic_pointer_cast<IndexMapExpr>(raw_src_idxs[i]->expr);
      if (!idxs) {
        throw std::runtime_error("oops: src_idxs");
      }
      expr->srcs.emplace_back(idxs);
      if (use_mlir()) {
        src_values.emplace_back(raw_src_idxs[i]->value);
      }
    }
    expr->ComputeShape(layout);
    if (use_mlir()) {
      auto value = GlobalContext::get()->MakeContractionOp(  //
          getAggregationKind(agg_op),                        //
          getCombinationKind(combo_op),                      //
          src_values,                                        //
          raw_sink_idxs->value,                              //
          raw_sink_sizes->value,                             //
          name);
      return new plaidml_expr{expr, value};
    }
    return new plaidml_expr{expr};
  });
}

void plaidml_expr_contraction_add_constraint(  //
    plaidml_error* err,                        //
    plaidml_expr* expr,                        //
    plaidml_poly_expr* lhs,                    //
    plaidml_dim_expr* rhs) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_contraction_add_constraint");
    if (!expr) {
      throw std::runtime_error("add_constraint can only be specified on a contraction.");
    }
    if (use_mlir()) {
      GlobalContext::get()->AddConstraint(expr->value, lhs->value, rhs->value);
    }
    auto cion = std::dynamic_pointer_cast<ContractionExpr>(expr->expr);
    if (!cion) {
      throw std::runtime_error("add_constraint can only be specified on a contraction.");
    }
    auto constraint = std::make_shared<ConstraintExpr>(lhs->expr, rhs->expr);
    cion->constraints.emplace_back(constraint);
  });
}

void plaidml_expr_contraction_set_no_defract(  //
    plaidml_error* err,                        //
    plaidml_expr* expr,                        //
    bool no_defract) {
  // TODO(MLIR)
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_contraction_set_no_defract");
    if (!expr) {
      throw std::runtime_error("no_defract can only be specified on a contraction.");
    }
    auto cion = std::dynamic_pointer_cast<ContractionExpr>(expr->expr);
    if (!cion) {
      throw std::runtime_error("no_defract can only be specified on a contraction.");
    }
    cion->no_defract = no_defract;
  });
}

void plaidml_expr_contraction_set_use_default(  //
    plaidml_error* err,                         //
    plaidml_expr* expr,                         //
    plaidml_expr* use_default) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_contraction_set_use_default");
    if (!expr) {
      throw std::runtime_error("use_default can only be specified on a contraction.");
    }
    if (use_mlir()) {
      GlobalContext::get()->SetUseDefault(expr->value, use_default->value);
    }
    auto cion = std::dynamic_pointer_cast<ContractionExpr>(expr->expr);
    if (!cion) {
      throw std::runtime_error("use_default can only be specified on a contraction.");
    }
    cion->use_default = use_default->expr;
  });
}

void plaidml_expr_gradient(  //
    plaidml_error* err,      //
    size_t nwrts,            //
    plaidml_expr** wrts,     //
    plaidml_expr* loss,      //
    plaidml_expr** derivs) {
  // Given a forward pass tensor operation that takes `nwrt` inputs given in
  // `wrt` and produces the output `loss`, produce the derivatives for each
  // tensor in `wrt` and store these `nwrt` derivatives in `derivs` in the
  // corresponding order as they were received in `wrt`.
  // TODO(MLIR)
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_gradient");
    std::vector<ExprPtr> wrt_exprs(nwrts);
    std::vector<mlir::Value*> wrt_values(nwrts);
    for (size_t i = 0; i < nwrts; i++) {
      wrt_exprs[i] = wrts[i]->expr;
      wrt_values[i] = wrts[i]->value;
    }
    auto deriv_exprs = ComputeGradients(wrt_exprs, loss->expr);
    if (use_mlir()) {
      auto deriv_values = GlobalContext::get()->ComputeGradients(wrt_values, loss->value);
      for (size_t i = 0; i < nwrts; i++) {
        derivs[i] = new plaidml_expr{deriv_exprs[i], deriv_values[i]};
      }
    } else {
      for (size_t i = 0; i < nwrts; i++) {
        derivs[i] = new plaidml_expr{deriv_exprs[i]};
      }
    }
  });
}

void plaidml_deriv_register(  //
    plaidml_error* err,       //
    const char* name,         //
    plaidml_deriv fn,         //
    void* user_ctx) {
  // TODO
  auto thunk = [](const ExprPtr& Y,                //
                  const ExprPtr& dY,               //
                  const std::vector<ExprPtr>& Xs,  //
                  void* user_fn,                   //
                  void* user_ctx) {
    IVLOG(6, "plaidml_deriv_register> thunk");
    plaidml_deriv fn = reinterpret_cast<plaidml_deriv>(user_fn);
    std::vector<plaidml_expr*> X_exprs(Xs.size());
    std::vector<plaidml_expr*> dX_exprs(Xs.size());
    auto Y_expr = new plaidml_expr{Y};
    auto dY_expr = new plaidml_expr{dY};
    for (size_t i = 0; i < X_exprs.size(); i++) {
      X_exprs[i] = new plaidml_expr{Xs[i]};
    }
    fn(user_ctx, Y_expr, dY_expr, X_exprs.size(), X_exprs.data(), dX_exprs.data());
    std::vector<ExprPtr> ret(Xs.size());
    for (size_t i = 0; i < ret.size(); i++) {
      ret[i] = dX_exprs[i]->expr;
      delete dX_exprs[i];
    }
    return ret;
  };
  ffi_wrap_void(err, [&] {
    IVLOG(5, "plaidml_deriv_register> " << name);
    DerivRegistry::Instance()->Register(name, thunk, reinterpret_cast<void*>(fn), user_ctx);
  });
}

void plaidml_poly_expr_free(plaidml_error* err, plaidml_poly_expr* expr) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_poly_expr_free> " << expr->expr->str());
    if (expr->value) {
      GlobalContext::get()->Destroy(expr->value);
    }
    delete expr;
  });
}

plaidml_string* plaidml_poly_expr_repr(  //
    plaidml_error* err,                  //
    plaidml_poly_expr* expr) {
  // TODO(MLIR)
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {  //
    IVLOG(3, "plaidml_poly_expr_repr");
    return new plaidml_string{expr->expr->str()};
  });
}

PLAIDML_EDSL_API plaidml_poly_expr* plaidml_poly_expr_dim(  //
    plaidml_error* err,                                     //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {  //
    IVLOG(3, "plaidml_poly_expr_dim");
    if (use_mlir()) {
      return new plaidml_poly_expr{std::make_shared<PolyDimExpr>(expr->expr), expr->value};
    } else {
      return new plaidml_poly_expr{std::make_shared<PolyDimExpr>(expr->expr)};
    }
  });
}

plaidml_poly_expr* plaidml_poly_expr_index(  //
    plaidml_error* err,                      //
    const char* name) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_index");
    if (use_mlir()) {
      auto value = GlobalContext::get()->MakeAffineIndexOp(name);
      return new plaidml_poly_expr{std::make_shared<PolyIndex>(next_idx_id++, std::string{name}), value};
    } else {
      return new plaidml_poly_expr{std::make_shared<PolyIndex>(next_idx_id++, std::string{name})};
    }
  });
}

plaidml_poly_expr* plaidml_poly_expr_literal(  //
    plaidml_error* err,                        //
    int64_t value) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_literal> " << value);
    if (use_mlir()) {
      auto mlir_value = GlobalContext::get()->MakeAffineConstantOp(value);
      return new plaidml_poly_expr{std::make_shared<PolyLiteral>(value), mlir_value};
    } else {
      return new plaidml_poly_expr{std::make_shared<PolyLiteral>(value)};
    }
  });
}

plaidml_poly_expr* plaidml_poly_expr_op(  //
    plaidml_error* err,                   //
    plaidml_int_op op,                    //
    size_t nargs,                         //
    plaidml_poly_expr** args) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_op> " << op);
    std::vector<std::shared_ptr<PolyExpr>> vec_args(nargs);
    std::vector<mlir::Value*> values(nargs);
    for (size_t i = 0; i < nargs; i++) {
      vec_args[i] = args[i]->expr;
      values[i] = args[i]->value;
      IVLOG(3, "  arg: " << values[i] << ": " << vec_args[i]->str());
    }
    if (use_mlir()) {
      auto value = MakeAffineOp(op, values);
      return new plaidml_poly_expr{MakeOp(static_cast<IntOp>(op), vec_args), value};
    } else {
      return new plaidml_poly_expr{MakeOp(static_cast<IntOp>(op), vec_args)};
    }
  });
}

void plaidml_dim_expr_free(  //
    plaidml_error* err,      //
    plaidml_dim_expr* expr) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_dim_expr_free> " << expr->expr->str());
    if (expr->value) {
      GlobalContext::get()->Destroy(expr->value);
    }
    delete expr;
  });
}

plaidml_string* plaidml_dim_expr_repr(  //
    plaidml_error* err,                 //
    plaidml_dim_expr* expr) {
  // TODO(MLIR)
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {  //
    IVLOG(3, "plaidml_dim_expr_repr");
    return new plaidml_string{expr->expr->str()};
  });
}

plaidml_dim_expr* plaidml_dim_expr_none(  //
    plaidml_error* err                    //
) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {  //
    IVLOG(3, "plaidml_dim_expr_none");
    if (use_mlir()) {
      auto value = GlobalContext::get()->MakeNoneOp();
      return new plaidml_dim_expr{std::make_shared<DimNoneExpr>(), value};
    } else {
      return new plaidml_dim_expr{std::make_shared<DimNoneExpr>()};
    }
  });
}

// plaidml_dim_expr* plaidml_dim_expr_ref(  //
//     plaidml_error* err,                  //
//     plaidml_expr* ref,                   //
//     size_t dim) {
//   return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
//     IVLOG(3, "plaidml_dim_expr_ref");
//     auto value = GlobalContext::get()->MakeDimOp(ref->value, dim);
//     return new plaidml_dim_expr{std::make_shared<DimRefExpr>(ref->expr, dim), value};
//   });
// }

plaidml_dim_expr* plaidml_dim_expr_int(  //
    plaidml_error* err,                  //
    int64_t value) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_int> " << value);
    if (use_mlir()) {
      auto mlir_value = GlobalContext::get()->MakeAffineConstantOp(value);
      return new plaidml_dim_expr{std::make_shared<DimIntExpr>(value), mlir_value};
    } else {
      return new plaidml_dim_expr{std::make_shared<DimIntExpr>(value)};
    }
  });
}

int64_t plaidml_dim_expr_get_int(  //
    plaidml_error* err,            //
    plaidml_dim_expr* expr) {
  // TODO(MLIR)
  return ffi_wrap<int64_t>(err, 0, [&] {
    IVLOG(3, "plaidml_dim_expr_get_int");
    if (!expr) {
      throw std::runtime_error("plaidml_dim_expr_get_int can only be used on an integer value");
    }
    auto int_expr = std::dynamic_pointer_cast<DimIntExpr>(expr->expr);
    if (!int_expr) {
      throw std::runtime_error("plaidml_dim_expr_get_int can only be used on an integer value");
    }
    return int_expr->value;
  });
}

plaidml_dim_expr* plaidml_dim_expr_op(  //
    plaidml_error* err,                 //
    plaidml_int_op op,                  //
    size_t nargs,                       //
    plaidml_dim_expr** args) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_op> " << op);
    std::vector<DimExprPtr> vec_args(nargs);
    std::vector<mlir::Value*> values(nargs);
    for (size_t i = 0; i < nargs; i++) {
      vec_args[i] = args[i]->expr;
      values[i] = args[i]->value;
      IVLOG(3, "  arg: " << values[i] << ": " << vec_args[i]->str());
    }
    if (use_mlir()) {
      auto value = MakeAffineOp(op, values);
      return new plaidml_dim_expr{MakeOp(static_cast<IntOp>(op), vec_args), value};
    } else {
      return new plaidml_dim_expr{MakeOp(static_cast<IntOp>(op), vec_args)};
    }
  });
}

void plaidml_program_free(  //
    plaidml_error* err,     //
    plaidml_program* program) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_program_free");
    delete program;
  });
}

plaidml_program* plaidml_program_evaluate(  //
    plaidml_error* err,                     //
    const char* name,                       //
    size_t noutputs,                        //
    plaidml_expr** raw_outputs,             //
    plaidml_expr** new_outputs,             //
    size_t nupdates,                        //
    plaidml_expr** src_updates,             //
    plaidml_expr** dst_updates) {
  return ffi_wrap<plaidml_program*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_program_evaluate");
    bool has_every_value = true;
    ProgramMutations mutations;
    std::vector<ExprPtr> outputs(noutputs);
    std::vector<mlir::Value*> values(noutputs);
    for (size_t i = 0; i < noutputs; i++) {
      if (!raw_outputs[i]) {
        throw std::runtime_error("Undefined output in plaidml_program_evaluate");
      }
      mutations.outputs.emplace_back(raw_outputs[i]->expr);
      values[i] = raw_outputs[i]->value;
      if (!values[i]) {
        IVLOG(5, "Found a missing value! (index " << i << ")");
        has_every_value = false;
      }
    }
    std::vector<ProgramUpdate> updates(nupdates);
    for (size_t i = 0; i < nupdates; i++) {
      if (!src_updates[i]) {
        throw std::runtime_error("Undefined update src in plaidml_program_evaluate");
      }
      if (!dst_updates[i]) {
        throw std::runtime_error("Undefined update dst in plaidml_program_evaluate");
      }
      mutations.updates.emplace_back(ProgramUpdate{src_updates[i]->expr, dst_updates[i]->expr});
    }
    // TODO(MLIR): updates
    if (has_every_value && use_mlir()) {
      std::vector<mlir::Value*> new_values(noutputs);
      auto program = GlobalContext::get()->MakeProgram(name, values, new_values);
      auto ret = new plaidml_program{Evaluate(name, mutations), program};
      if (noutputs != ret->eval.outputs.size()) {
        throw std::runtime_error("Internal error: noutputs != ret->eval.outputs.size()");
      }
      for (size_t i = 0; i < noutputs; i++) {
        new_outputs[i] = new plaidml_expr{ret->eval.outputs[i], new_values[i]};
      }
      return ret;
    } else {
      auto ret = new plaidml_program{Evaluate(name, mutations)};
      if (noutputs != ret->eval.outputs.size()) {
        throw std::runtime_error("Internal error: noutputs != ret->eval.outputs.size()");
      }
      for (size_t i = 0; i < noutputs; i++) {
        new_outputs[i] = new plaidml_expr{ret->eval.outputs[i]};
      }
      return ret;
    }
  });
}

plaidml_string* plaidml_program_repr(  //
    plaidml_error* err,                //
    plaidml_program* program) {
  // TODO(MLIR)
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {  //
    IVLOG(3, "plaidml_program_repr");
    return new plaidml_string{to_string(program->eval.runinfo.program)};
  });
}

const void* plaidml_program_runinfo(  //
    plaidml_error* err,               //
    plaidml_program* program) {
  // TODO(MLIR)
  return ffi_wrap<const void*>(err, nullptr, [&] {  //
    IVLOG(3, "plaidml_program_runinfo");
    return &program->eval.runinfo;
  });
}

}  // extern "C"
