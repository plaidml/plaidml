// Copyright 2019 Intel Corporation.

#include "plaidml2/edsl/ffi.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <boost/format.hpp>

#include "base/util/logging.h"
#include "plaidml2/core/internal.h"
#include "plaidml2/edsl/derivs.h"
#include "pmlc/dialect/scalar/ops.h"
#include "tile/lang/ast/ast.h"
#include "tile/lang/ast/gradient.h"

using namespace vertexai::tile;             // NOLINT
using namespace vertexai::tile::lang;       // NOLINT
using namespace vertexai::tile::lang::ast;  // NOLINT
using namespace pmlc::dialect::scalar;      // NOLINT

using mlir::Type;
using mlir::edsc::ValueHandle;
using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;

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

}  // namespace

extern "C" {

struct plaidml_logical_shape {
  LogicalShape shape;
};

struct plaidml_dim_expr {
  DimExprPtr expr;
};

struct plaidml_poly_expr {
  PolyExprPtr expr;
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

void plaidml_expr_free(  //
    plaidml_error* err,  //
    plaidml_expr* expr) {
  ffi_wrap_void(err, [&] {  //
    delete expr;
  });
}

plaidml_logical_shape* plaidml_expr_get_shape(  //
    plaidml_error* err,                         //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_logical_shape*>(err, nullptr, [&] {  //
    return new plaidml_logical_shape{expr->expr->shape};
  });
}

void plaidml_expr_bind_shape(  //
    plaidml_error* err,        //
    plaidml_expr* expr,        //
    plaidml_logical_shape* shape) {
  return ffi_wrap_void(err, [&] {  //
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
    std::vector<DimExprPtr> vec_dims(ndims);
    for (size_t i = 0; i < ndims; i++) {
      vec_dims[i] = dims[i]->expr;
    }
    expr->expr->shape.bind_dims(&vec_dims);
    for (size_t i = 0; i < ndims; i++) {
      dims[i]->expr = vec_dims[i];
    }
  });
}

plaidml_string* plaidml_expr_repr(  //
    plaidml_error* err,             //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {  //
    return new plaidml_string{expr->expr->str()};
  });
}

plaidml_expr* plaidml_expr_dim(  //
    plaidml_error* err,          //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {  //
    return new plaidml_expr{std::make_shared<DimExprExpr>(expr->expr)};
  });
}

plaidml_expr* plaidml_expr_param(  //
    plaidml_error* err,            //
    plaidml_logical_shape* shape,  //
    plaidml_buffer* buffer,        //
    const char* name) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    auto expr = std::make_shared<ParamExpr>(name);
    if (buffer) {
      expr->buffer = buffer->buffer;
    }
    expr->ComputeShape(expr, shape->shape);
    return new plaidml_expr{expr};
  });
}

plaidml_expr* plaidml_expr_clone(  //
    plaidml_error* err,            //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {  //
    return new plaidml_expr{expr->expr};
  });
}
plaidml_expr_kind plaidml_expr_get_kind(  //
    plaidml_error* err,                   //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_expr_kind>(err, PLAIDML_EXPR_NONE, [&] {
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
    return PLAIDML_EXPR_TENSOR;
  });
}

plaidml_expr* plaidml_expr_none(  //
    plaidml_error* err            //
) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {  //
    return new plaidml_expr{std::make_shared<NoneExpr>()};
  });
}

plaidml_expr* plaidml_expr_tuple(  //
    plaidml_error* err,            //
    size_t nargs,                  //
    plaidml_expr** args) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    std::vector<ExprPtr> exprs(nargs);
    for (size_t i = 0; i < nargs; i++) {
      exprs[i] = args[i]->expr;
    }
    return new plaidml_expr{std::make_shared<TupleExpr>(exprs)};
  });
}

size_t plaidml_expr_tuple_get_count(  //
    plaidml_error* err,               //
    plaidml_expr* expr) {
  return ffi_wrap<size_t>(err, 0, [&] {
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
    auto tuple_expr = std::dynamic_pointer_cast<TupleExpr>(expr->expr);
    if (!tuple_expr) {
      throw std::runtime_error("Expression is not a tuple");
    }
    for (size_t i = 0; i < std::min(nexprs, tuple_expr->exprs.size()); i++) {
      exprs[i] = new plaidml_expr{tuple_expr->exprs[i]};
    }
  });
}

plaidml_expr* plaidml_expr_str(  //
    plaidml_error* err,          //
    const char* value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {  //
    return new plaidml_expr{std::make_shared<StringExpr>(value)};
  });
}

plaidml_string* plaidml_expr_str_get_value(  //
    plaidml_error* err,                      //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
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
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {  //
    return new plaidml_expr{std::make_shared<IntConst>(value)};
  });
}

int64_t plaidml_expr_int_get_value(  //
    plaidml_error* err,              //
    plaidml_expr* expr) {
  return ffi_wrap<int64_t>(err, 0, [&] {
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
    // ScalarConstantOp
    // auto fp_type = Type::getFromOpaquePointer(type.type).cast<FloatType>();
    // auto fp_value = llvm::APFloat(value);
    // bool lostPrecision;
    // value.convert(floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven, &lostPrecision);
    // auto std_f32 = FloatType::getF32(globalContext());
    // auto f32 = ScalarType::get(globalContext(), DataType::FLOAT32);
    // auto x2 = builder.create<ScalarConstantOp>(loc, f32, builder.getFloatAttr(std_f32, 1));
    return new plaidml_expr{
        std::make_shared<FloatConst>(value),
        std::make_shared<ValueHandle>(ValueHandle::create<ScalarConstantOp>(value)),
    };
  });
}

double plaidml_expr_float_get_value(  //
    plaidml_error* err,               //
    plaidml_expr* expr) {
  return ffi_wrap<double>(err, 0, [&] {
    // auto value = expr->value->getValue();
    // value->dump();
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
    std::vector<ExprPtr> exprs(nargs);
    for (size_t i = 0; i < nargs; i++) {
      if (!args[i]) {
        throw std::runtime_error(str(boost::format("Undefined tensor in call to %1%()") % fn));
      }
      exprs[i] = args[i]->expr;
    }
    return new plaidml_expr{MakeCall(fn, exprs)};
  });
}

plaidml_expr* plaidml_expr_tensor_spec(  //
    plaidml_error* err,                  //
    plaidml_expr* ref,                   //
    size_t ndims,                        //
    plaidml_poly_expr** input_idxs,      //
    plaidml_dim_expr** output_dims) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    std::vector<DimExprPtr> vec_dims;
    std::vector<std::shared_ptr<PolyExpr>> vec_idxs(ndims);
    for (size_t i = 0; i < ndims; i++) {
      vec_idxs[i] = input_idxs[i]->expr;
      if (output_dims) {
        if (!output_dims[i]) {
          throw std::runtime_error("Undefined output_dim in TensorSpec");
        }
        vec_dims.emplace_back(output_dims[i]->expr);
      }
    }
    if (ref) {
      return new plaidml_expr{std::make_shared<TensorSpecExpr>(ref->expr, vec_idxs)};
    }
    return new plaidml_expr{std::make_shared<TensorSpecExpr>(vec_idxs, vec_dims)};
  });
}

plaidml_expr* plaidml_expr_contraction(  //
    plaidml_error* err,                  //
    plaidml_agg_op agg_op,               //
    plaidml_combo_op combo_op,           //
    plaidml_expr* raw_output,            //
    size_t ninputs,                      //
    plaidml_expr** raw_inputs,           //
    const char* name,                    //
    const char* layout) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
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
    expr->ComputeShape(layout);
    return new plaidml_expr{expr};
  });
}

void plaidml_expr_contraction_set_no_defract(  //
    plaidml_error* err,                        //
    plaidml_expr* expr,                        //
    bool no_defract) {
  ffi_wrap_void(err, [&] {
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
  // Given a forward pass tensor operation that takes `nwrt` inputs given in `wrt` and produces the output `loss`,
  // produce the derivatives for each tensor in `wrt` and store these `nwrt` derivatives in `derivs` in the
  // corresponding order as they were received in `wrt`.
  ffi_wrap_void(err, [&] {
    std::vector<ExprPtr> wrt_exprs(nwrts);
    for (size_t i = 0; i < nwrts; i++) {
      wrt_exprs[i] = wrts[i]->expr;
    }
    auto deriv_exprs = ComputeGradients(wrt_exprs, loss->expr);
    for (size_t i = 0; i < nwrts; i++) {
      derivs[i] = new plaidml_expr{deriv_exprs[i]};
    }
  });
}

void plaidml_deriv_register(  //
    plaidml_error* err,       //
    const char* name,         //
    plaidml_deriv fn,         //
    void* user_ctx) {
  auto thunk = [](const ExprPtr& Y,                //
                  const ExprPtr& dY,               //
                  const std::vector<ExprPtr>& Xs,  //
                  void* user_fn,                   //
                  void* user_ctx) {
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
  ffi_wrap_void(err, [&] {  //
    delete expr;
  });
}

plaidml_string* plaidml_poly_expr_repr(  //
    plaidml_error* err,                  //
    plaidml_poly_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {  //
    return new plaidml_string{expr->expr->str()};
  });
}

PLAIDML_EDSL_API plaidml_poly_expr* plaidml_poly_expr_dim(  //
    plaidml_error* err,                                     //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {  //
    return new plaidml_poly_expr{std::make_shared<PolyDimExpr>(expr->expr)};
  });
}

plaidml_poly_expr* plaidml_poly_expr_index(  //
    plaidml_error* err,                      //
    const char* name) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {  //
    return new plaidml_poly_expr{std::make_shared<PolyIndex>(next_idx_id++, std::string{name})};
  });
}

plaidml_poly_expr* plaidml_poly_expr_literal(  //
    plaidml_error* err,                        //
    int64_t value) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {  //
    return new plaidml_poly_expr{std::make_shared<PolyLiteral>(value)};
  });
}

plaidml_poly_expr* plaidml_poly_expr_op(  //
    plaidml_error* err,                   //
    plaidml_int_op op,                    //
    size_t nargs,                         //
    plaidml_poly_expr** args) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    std::vector<std::shared_ptr<PolyExpr>> vec_args(nargs);
    for (size_t i = 0; i < nargs; i++) {
      vec_args[i] = args[i]->expr;
    }
    return new plaidml_poly_expr{MakeOp(static_cast<IntOp>(op), vec_args)};
  });
}

void plaidml_poly_expr_add_constraint(  //
    plaidml_error* err,                 //
    plaidml_poly_expr* lhs,             //
    plaidml_dim_expr* rhs) {
  ffi_wrap_void(err, [&] {
    auto constraint = std::make_shared<ConstraintExpr>(lhs->expr, rhs->expr);
    ConstraintApplier applier(constraint);
    lhs->expr->Accept(&applier);
  });
}

void plaidml_dim_expr_free(  //
    plaidml_error* err,      //
    plaidml_dim_expr* expr) {
  ffi_wrap_void(err, [&] {  //
    delete expr;
  });
}

plaidml_string* plaidml_dim_expr_repr(  //
    plaidml_error* err,                 //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {  //
    return new plaidml_string{expr->expr->str()};
  });
}

plaidml_dim_expr* plaidml_dim_expr_none(  //
    plaidml_error* err                    //
) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {  //
    return new plaidml_dim_expr{std::make_shared<DimNoneExpr>()};
  });
}

plaidml_dim_expr* plaidml_dim_expr_ref(  //
    plaidml_error* err,                  //
    plaidml_expr* ref,                   //
    size_t dim) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {  //
    return new plaidml_dim_expr{std::make_shared<DimRefExpr>(ref->expr, dim)};
  });
}

plaidml_dim_expr* plaidml_dim_expr_int(  //
    plaidml_error* err,                  //
    int64_t value) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {  //
    return new plaidml_dim_expr{std::make_shared<DimIntExpr>(value)};
  });
}

int64_t plaidml_dim_expr_get_int(  //
    plaidml_error* err,            //
    plaidml_dim_expr* expr) {
  return ffi_wrap<int64_t>(err, 0, [&] {  //
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
    std::vector<DimExprPtr> vec_args(nargs);
    for (size_t i = 0; i < nargs; i++) {
      vec_args[i] = args[i]->expr;
    }
    return new plaidml_dim_expr{MakeOp(static_cast<IntOp>(op), vec_args)};
  });
}

void plaidml_program_free(  //
    plaidml_error* err,     //
    plaidml_program* program) {
  ffi_wrap_void(err, [&] {  //
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
    ProgramMutations mutations;
    std::vector<ExprPtr> outputs(noutputs);
    for (size_t i = 0; i < noutputs; i++) {
      if (!raw_outputs[i]) {
        throw std::runtime_error("Undefined output in plaidml_program_evaluate");
      }
      mutations.outputs.emplace_back(raw_outputs[i]->expr);
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
    auto ret = new plaidml_program{Evaluate(name, mutations)};
    if (noutputs != ret->eval.outputs.size()) {
      throw std::runtime_error("Internal error: noutputs != ret->eval.outputs.size()");
    }
    for (size_t i = 0; i < noutputs; i++) {
      new_outputs[i] = new plaidml_expr{ret->eval.outputs[i]};
    }
    return ret;
  });
}

plaidml_string* plaidml_program_repr(  //
    plaidml_error* err,                //
    plaidml_program* program) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {  //
    return new plaidml_string{to_string(program->eval.runinfo.program)};
  });
}

const void* plaidml_program_runinfo(  //
    plaidml_error* err,               //
    plaidml_program* program) {
  return ffi_wrap<const void*>(err, nullptr, [&] {  //
    return &program->eval.runinfo;
  });
}

}  // extern "C"
