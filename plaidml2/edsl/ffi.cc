// Copyright 2019 Intel Corporation.

#include "plaidml2/edsl/ffi.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "llvm/Support/FormatVariadic.h"

#include "base/util/logging.h"
#include "plaidml2/core/internal.h"
#include "plaidml2/edsl/derivs.h"

#ifdef PLAIDML_AST
#include "tile/lang/ast/ast.h"
#include "tile/lang/ast/gradient.h"
#endif

#ifdef PLAIDML_MLIR
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/eltwise/types.h"
#include "pmlc/dialect/tile/gradient.h"
#include "pmlc/dialect/tile/ops.h"
#include "pmlc/util/enums.h"
#endif

using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using plaidml::core::GlobalContext;
using vertexai::tile::DataType;

#ifdef PLAIDML_AST
using namespace vertexai::tile;             // NOLINT
using namespace vertexai::tile::lang;       // NOLINT
using namespace vertexai::tile::lang::ast;  // NOLINT
#endif

#ifdef PLAIDML_MLIR
using pmlc::dialect::eltwise::ScalarType;
using pmlc::dialect::tile::DerivRegistry;
using pmlc::dialect::tile::ProgramMutations;
using pmlc::dialect::tile::ProgramUpdate;
using pmlc::dialect::tile::TileBuilder;
using pmlc::util::AggregationKind;
using pmlc::util::CombinationKind;
#endif

namespace {

#ifdef PLAIDML_AST
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
#endif

#ifdef PLAIDML_MLIR
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
#endif

}  // namespace

extern "C" {

struct plaidml_logical_shape {
#ifdef PLAIDML_AST
  LogicalShape shape;
#endif
#ifdef PLAIDML_MLIR
  mlir::RankedTensorType type;
#endif
};

struct plaidml_poly_expr {
#ifdef PLAIDML_AST
  PolyExprPtr expr;
#endif
#ifdef PLAIDML_MLIR
  mlir::Value* value = nullptr;
#endif
};

void plaidml_edsl_init(  //
    plaidml_error* err) {
  static std::once_flag is_initialized;
  ffi_wrap_void(err, [&] {
    std::call_once(is_initialized, []() {
      IVLOG(1, "plaidml_edsl_init");
      plaidml::edsl::RegisterDerivs();
    });
  });
}

plaidml_logical_shape* plaidml_logical_shape_alloc(  //
    plaidml_error* err,                              //
    plaidml_datatype dtype,                          //
    size_t ndims,                                    //
    const int64_t* dims) {
  return ffi_wrap<plaidml_logical_shape*>(err, nullptr, [&] {
#ifdef PLAIDML_AST
    auto ret = new plaidml_logical_shape;
    ret->shape.dtype = static_cast<DataType>(dtype);
    for (size_t i = 0; i < ndims; i++) {
      auto int_expr = std::make_shared<DimIntExpr>(dims[i]);
      ret->shape.dims.emplace_back(LogicalDim{int_expr});
    }
    return ret;
#endif
#ifdef PLAIDML_MLIR
    llvm::SmallVector<int64_t, 6> dimsVec;
    for (size_t i = 0; i < ndims; i++) {
      dimsVec.emplace_back(dims[i]);
    }
    auto ret = new plaidml_logical_shape;
    ret->type = GlobalContext::get()->MakeRankedTensorType(static_cast<DataType>(dtype), dimsVec);
    return ret;
#endif
  });
}

plaidml_logical_shape* plaidml_logical_shape_clone(  //
    plaidml_error* err,                              //
    plaidml_logical_shape* shape) {
  return ffi_wrap<plaidml_logical_shape*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_logical_shape");
#ifdef PLAIDML_AST
    return new plaidml_logical_shape{shape->shape};
#endif
#ifdef PLAIDML_MLIR
    // TODO(MLIR): deal with clone
    return new plaidml_logical_shape{shape->type};
#endif
  });
}

plaidml_string* plaidml_logical_shape_repr(  //
    plaidml_error* err,                      //
    plaidml_logical_shape* shape) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
#ifdef PLAIDML_AST
    std::stringstream ss;
    ss << shape->shape.str();
    return new plaidml_string{ss.str()};
#endif
#ifdef PLAIDML_MLIR
    return new plaidml_string{mlir::debugString(shape->type)};
#endif
  });
}

size_t plaidml_logical_shape_get_ndims(  //
    plaidml_error* err,                  //
    plaidml_logical_shape* shape) {
  return ffi_wrap<size_t>(err, 0, [&] {
#ifdef PLAIDML_AST
    return shape->shape.dims.size();
#endif
#ifdef PLAIDML_MLIR
    return shape->type.getRank();
#endif
  });
}

plaidml_datatype plaidml_logical_shape_get_dtype(  //
    plaidml_error* err,                            //
    plaidml_logical_shape* shape) {
  return ffi_wrap<plaidml_datatype>(err, PLAIDML_DATA_INVALID, [&] {
#ifdef PLAIDML_AST
    return static_cast<plaidml_datatype>(shape->shape.dtype);
#endif
#ifdef PLAIDML_MLIR
    auto elementType = shape->type.getElementType();
    auto scalarType = elementType.dyn_cast<ScalarType>();
    if (!scalarType) {
      throw std::runtime_error("Expected scalar type");
    }
    return static_cast<plaidml_datatype>(scalarType.type());
#endif
  });
}

int64_t plaidml_logical_shape_get_dim_int(  //
    plaidml_error* err,                     //
    plaidml_logical_shape* shape,           //
    size_t dim) {
  return ffi_wrap<int64_t>(err, 0, [&] {
#ifdef PLAIDML_AST
    auto dim_expr = shape->shape.dims.at(dim).expr;
    auto int_expr = std::dynamic_pointer_cast<DimIntExpr>(dim_expr);
    if (int_expr) {
      return int_expr->value;
    }
    return static_cast<int64_t>(0);
#endif
#ifdef PLAIDML_MLIR
    const auto& dims = shape->type.getShape();
    if (dims.size() < dim) {
      throw std::range_error("Index out of range");
    }
    auto ret = dims[dim];
    if (ret < 0) {
      return static_cast<int64_t>(0);
    }
    return ret;
#endif
  });
}

plaidml_dim_expr* plaidml_logical_shape_get_dim_expr(  //
    plaidml_error* err,                                //
    plaidml_logical_shape* shape,                      //
    size_t dim) {
  return ffi_wrap<plaidml_dim_expr*>(err, 0, [&] {  //
#ifdef PLAIDML_AST
    return new plaidml_dim_expr{shape->shape.dims.at(dim).expr};
#endif
#ifdef PLAIDML_MLIR
    throw std::runtime_error("NYI: plaidml_logical_shape_get_dim_expr");
    return nullptr;
#endif
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
  return ffi_wrap<plaidml_shape*>(err, 0, [&] {
#ifdef PLAIDML_AST
    return new plaidml_shape{IntoTensorShape(shape->shape)};
#endif
#ifdef PLAIDML_MLIR
    return new plaidml_shape{GlobalContext::get()->IntoTensorType(shape->type)};
#endif
  });
}

void plaidml_expr_free(  //
    plaidml_error* err,  //
    plaidml_expr* expr) {
  ffi_wrap_void(err, [&] {
#ifdef PLAIDML_AST
    IVLOG(3, "plaidml_expr_free> " << expr->expr->str());
#endif
#ifdef PLAIDML_MLIR
    IVLOG(3, "plaidml_expr_free> " << expr->value);
    IVLOG(3, "plaidml_expr_free> " << mlir::debugString(*expr->value));
    GlobalContext::get()->Destroy(expr->value);
#endif
    delete expr;
  });
}

void* plaidml_expr_ptr(  //
    plaidml_error* err,  //
    plaidml_expr* expr) {
  return ffi_wrap<void*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_ptr");
#ifdef PLAIDML_AST
    return expr->expr.get();
#endif
#ifdef PLAIDML_MLIR
    return expr->value;
#endif
  });
}

plaidml_logical_shape* plaidml_expr_get_shape(  //
    plaidml_error* err,                         //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_logical_shape*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_get_shape");
    if (!expr) {
      throw std::runtime_error(
          "Cannot compute shape of null expr. Perhaps you requested the shape of an unassigned tensor?");
    }
#ifdef PLAIDML_AST
    return new plaidml_logical_shape{expr->expr->shape};
#endif
#ifdef PLAIDML_MLIR
    return new plaidml_logical_shape{GlobalContext::get()->ComputeShape(expr->value)};
#endif
  });
}

void plaidml_expr_bind_shape(  //
    plaidml_error* err,        //
    plaidml_expr* expr,        //
    plaidml_logical_shape* shape) {
  return ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_bind_shape");
#ifdef PLAIDML_AST
    auto param_expr = std::dynamic_pointer_cast<ParamExpr>(expr->expr);
    if (!param_expr) {
      throw std::runtime_error("Shape binding is only supported on ParamExprs");
    }
    param_expr->shape = shape->shape;
#endif
#ifdef PLAIDML_MLIR
    GlobalContext::get()->BindShape(expr->value, shape->type);
#endif
  });
}

void plaidml_expr_bind_dims(  //
    plaidml_error* err,       //
    plaidml_expr* expr,       //
    size_t ndims,             //
    plaidml_dim_expr** dims) {
  return ffi_wrap_void(err, [&] {
#ifdef PLAIDML_AST
    IVLOG(3, "plaidml_expr_bind_dims> " << expr->expr->str());
    std::vector<DimExprPtr> vec_dims(ndims);
    for (size_t i = 0; i < ndims; i++) {
      vec_dims[i] = dims[i]->expr;
    }
    expr->expr->shape.bind_dims(&vec_dims);
    for (size_t i = 0; i < ndims; i++) {
      dims[i]->expr = vec_dims[i];
    }
#endif
#ifdef PLAIDML_MLIR
    IVLOG(3, "plaidml_expr_bind_dims> " << mlir::debugString(*expr->value));
    llvm::SmallVector<mlir::Value**, 6> into;
    for (size_t i = 0; i < ndims; i++) {
      IVLOG(3, "bind_dims> i: " << i << ", from: " << expr->value << ", into: " << dims[i]->value);
      into.emplace_back(&dims[i]->value);
    }
    GlobalContext::get()->BindTensorDims(expr->value, into);
#endif
  });
}

plaidml_string* plaidml_expr_repr(  //
    plaidml_error* err,             //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_repr");
#ifdef PLAIDML_AST
    return new plaidml_string{expr->expr->str()};
#endif
#ifdef PLAIDML_MLIR
    return new plaidml_string{mlir::debugString(*expr->value)};
#endif
  });
}

plaidml_expr* plaidml_expr_dim(  //
    plaidml_error* err,          //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_dim");
#ifdef PLAIDML_AST
    return new plaidml_expr{std::make_shared<DimExprExpr>(expr->expr)};
#endif
#ifdef PLAIDML_MLIR
    // TODO: clone?
    return new plaidml_expr{expr->value};
#endif
  });
}

plaidml_expr* plaidml_expr_placeholder(  //
    plaidml_error* err,                  //
    plaidml_logical_shape* shape,        //
    plaidml_buffer* buffer,              //
    const char* name) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_placeholder");
#ifdef PLAIDML_AST
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
    return new plaidml_expr{expr};
#endif
#ifdef PLAIDML_MLIR
    return new plaidml_expr{
        GlobalContext::get()->MakePlaceholderOp(shape->type, buffer ? buffer->buffer : nullptr, name)};
#endif
  });
}

void plaidml_expr_param_reset(  //
    plaidml_error* err,         //
    plaidml_expr* expr,         //
    plaidml_buffer* buffer) {
  return ffi_wrap_void(err, [&] {
#ifdef PLAIDML_AST
    auto param_expr = std::dynamic_pointer_cast<ParamExpr>(expr->expr);
    if (param_expr) {
      param_expr->buffer = buffer->buffer;
    } else {
      throw std::runtime_error("ParamExpr value reset requested for non-ParamExpr");
    }
#endif
#ifdef PLAIDML_MLIR
    GlobalContext::get()->BindBuffer(expr->value, buffer->buffer);
#endif
  });
}

plaidml_expr* plaidml_expr_clone(  //
    plaidml_error* err,            //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
#ifdef PLAIDML_AST
    IVLOG(3, "plaidml_expr_clone> " << expr->expr->str());
    return new plaidml_expr{expr->expr};
#endif
#ifdef PLAIDML_MLIR
    IVLOG(3, "plaidml_expr_clone> " << mlir::debugString(*expr->value));
    // TODO(MLIR): deal with clone of expr->value
    return new plaidml_expr{expr->value};
#endif
  });
}

plaidml_dim_expr* plaidml_expr_get_dim(  //
    plaidml_error* err,                  //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
#ifdef PLAIDML_AST
    IVLOG(3, "plaidml_expr_get_dim> " << expr->expr->str());
    auto dim_expr = std::dynamic_pointer_cast<DimExprExpr>(expr->expr);
    if (!dim_expr) {
      throw std::runtime_error("plaidml_expr_get_dim can only be used on a DimExprExpr");
    }
    return new plaidml_dim_expr{dim_expr->expr};
#endif
#ifdef PLAIDML_MLIR
    IVLOG(3, "plaidml_expr_get_dim> " << mlir::debugString(*expr->value));
    // TODO(MLIR): deal with clone of expr->value
    return new plaidml_dim_expr{expr->value};
#endif
  });
}

plaidml_expr* plaidml_expr_int(  //
    plaidml_error* err,          //
    int64_t value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_int> " << value);
#ifdef PLAIDML_AST
    return new plaidml_expr{std::make_shared<IntConst>(value)};
#endif
#ifdef PLAIDML_MLIR
    return new plaidml_expr{GlobalContext::get()->MakeScalarConstantOp(value)};
#endif
  });
}

plaidml_expr* plaidml_expr_float(  //
    plaidml_error* err,            //
    double value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_float");
#ifdef PLAIDML_AST
    return new plaidml_expr{std::make_shared<FloatConst>(value)};
#endif
#ifdef PLAIDML_MLIR
    return new plaidml_expr{GlobalContext::get()->MakeScalarConstantOp(value)};
#endif
  });
}

plaidml_expr* plaidml_expr_cast(  //
    plaidml_error* err,           //
    plaidml_expr* tensor,         //
    plaidml_datatype dtype) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_cast");
#ifdef PLAIDML_AST
    static ExprPtr bits8 = std::make_shared<IntConst>(8);
    static ExprPtr bits16 = std::make_shared<IntConst>(16);
    static ExprPtr bits32 = std::make_shared<IntConst>(32);
    static ExprPtr bits64 = std::make_shared<IntConst>(64);
    switch (static_cast<DataType>(dtype)) {
      case DataType::INT8:
        return new plaidml_expr{MakeCall("as_int", {tensor->expr, bits8})};
      case DataType::INT16:
        return new plaidml_expr{MakeCall("as_int", {tensor->expr, bits16})};
      case DataType::INT32:
        return new plaidml_expr{MakeCall("as_int", {tensor->expr, bits32})};
      case DataType::INT64:
        return new plaidml_expr{MakeCall("as_int", {tensor->expr, bits64})};
      case DataType::UINT8:
        return new plaidml_expr{MakeCall("as_uint", {tensor->expr, bits8})};
      case DataType::UINT16:
        return new plaidml_expr{MakeCall("as_uint", {tensor->expr, bits16})};
      case DataType::UINT32:
        return new plaidml_expr{MakeCall("as_uint", {tensor->expr, bits32})};
      case DataType::UINT64:
        return new plaidml_expr{MakeCall("as_uint", {tensor->expr, bits64})};
      case DataType::FLOAT16:
        return new plaidml_expr{MakeCall("as_float", {tensor->expr, bits16})};
      case DataType::FLOAT32:
        return new plaidml_expr{MakeCall("as_float", {tensor->expr, bits32})};
      case DataType::FLOAT64:
        return new plaidml_expr{MakeCall("as_float", {tensor->expr, bits64})};
      default:
        throw std::runtime_error("Unsupported dtype for cast");
    }
#endif
#ifdef PLAIDML_MLIR
    return new plaidml_expr{GlobalContext::get()->MakeCastOp(tensor->value, static_cast<DataType>(dtype))};
#endif
  });
}

plaidml_expr* plaidml_expr_call(  //
    plaidml_error* err,           //
    const char* fn,               //
    size_t nargs,                 //
    plaidml_expr** args) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_call");
#ifdef PLAIDML_AST
    std::vector<ExprPtr> exprs(nargs);
    for (size_t i = 0; i < nargs; i++) {
      if (!args[i]) {
        throw std::runtime_error(str(boost::format("Undefined tensor in call to %1%()") % fn));
      }
      exprs[i] = args[i]->expr;
    }
    return new plaidml_expr{MakeCall(fn, exprs)};
#endif
#ifdef PLAIDML_MLIR
    std::vector<mlir::Value*> values(nargs);
    for (size_t i = 0; i < nargs; i++) {
      values[i] = args[i]->value;
    }
    return new plaidml_expr{GlobalContext::get()->MakePrimitiveOp(fn, values)};
#endif
  });
}

plaidml_expr* plaidml_expr_grad_override(  //
    plaidml_error* err,                    //
    plaidml_deriv fn,                      //
    void* user_ctx,                        //
    size_t nins,                           //
    plaidml_expr** ins,                    //
    plaidml_expr* out) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(5, "plaidml_grad_override");
#ifdef PLAIDML_AST
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
    auto deriv_entry = std::make_shared<ExprDerivEntry>();
    deriv_entry->fn = thunk;
    deriv_entry->user_fn = reinterpret_cast<void*>(fn);
    deriv_entry->user_ctx = user_ctx;
    std::vector<ExprPtr> in_exprs(nins);
    for (size_t i = 0; i < nins; i++) {
      if (!ins[i]) {
        throw std::runtime_error("Undefined input tensor in gradient override");
      }
      in_exprs[i] = ins[i]->expr;
    }
    if (!out) {
      throw std::runtime_error("Undefined output tensor in gradient override");
    }
    ExprPtr expr = MakeGradOverride(deriv_entry, in_exprs, out->expr);
    IVLOG(6, "The expr from plaidml_expr_grad_override has shape " << expr->shape.str());
    return new plaidml_expr{expr};
#endif
#ifdef PLAIDML_MLIR
    // TODO(MLIR)
    return new plaidml_expr{out->value};
#endif
  });
}

plaidml_expr* plaidml_expr_index_map(  //
    plaidml_error* err,                //
    plaidml_expr* ref,                 //
    size_t ndims,                      //
    plaidml_poly_expr** raw_idxs) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_index_map");
#ifdef PLAIDML_AST
    std::vector<std::shared_ptr<PolyExpr>> idx_exprs(ndims);
    for (size_t i = 0; i < ndims; i++) {
      idx_exprs[i] = raw_idxs[i]->expr;
    }
    if (ref) {
      return new plaidml_expr{std::make_shared<IndexMapExpr>(ref->expr, idx_exprs)};
    }
    return new plaidml_expr{std::make_shared<IndexMapExpr>(nullptr, idx_exprs)};
#endif
#ifdef PLAIDML_MLIR
    std::vector<mlir::Value*> idx_values(ndims);
    for (size_t i = 0; i < ndims; i++) {
      idx_values[i] = raw_idxs[i]->value;
    }
    if (ref) {
      return new plaidml_expr{GlobalContext::get()->MakeAffineSourceIndexMapOp(ref->value, idx_values)};
    }
    return new plaidml_expr{GlobalContext::get()->MakeAffineSinkIndexMapOp(idx_values)};
#endif
  });
}

plaidml_expr* plaidml_expr_size_map(  //
    plaidml_error* err,               //
    size_t ndims,                     //
    plaidml_dim_expr** raw_dims) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_size_map");
#ifdef PLAIDML_AST
    std::vector<DimExprPtr> dim_exprs;
    for (size_t i = 0; i < ndims; i++) {
      dim_exprs.emplace_back(raw_dims[i]->expr);
    }
    return new plaidml_expr{std::make_shared<SizeMapExpr>(dim_exprs)};
#endif
#ifdef PLAIDML_MLIR
    std::vector<mlir::Value*> dim_values;
    for (size_t i = 0; i < ndims; i++) {
      dim_values.emplace_back(raw_dims[i]->value);
    }
    return new plaidml_expr{GlobalContext::get()->MakeAffineSizeMapOp(dim_values)};
#endif
  });
}

plaidml_expr* plaidml_expr_contraction(  //
    plaidml_error* err,                  //
    plaidml_agg_op agg_op,               //
    plaidml_combo_op combo_op,           //
    plaidml_expr* sink_idxs,             //
    plaidml_expr* sink_sizes,            //
    size_t nsrcs,                        //
    plaidml_expr** src_idxs,             //
    const char* name) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_contraction");
#ifdef PLAIDML_AST
    auto expr = std::make_shared<ContractionExpr>();
    expr->name = name;
    expr->agg_op = into_agg_op(agg_op);
    expr->combo_op = into_combo_op(combo_op);
    expr->sink_idxs = std::dynamic_pointer_cast<IndexMapExpr>(sink_idxs->expr);
    if (!expr->sink_idxs) {
      throw std::runtime_error("oops: sink_idxs");
    }
    expr->sink_dims = std::dynamic_pointer_cast<SizeMapExpr>(sink_sizes->expr);
    if (!expr->sink_dims) {
      throw std::runtime_error("oops: sink_dims");
    }
    for (size_t i = 0; i < nsrcs; i++) {
      auto idxs = std::dynamic_pointer_cast<IndexMapExpr>(src_idxs[i]->expr);
      if (!idxs) {
        throw std::runtime_error("oops: src_idxs");
      }
      expr->srcs.emplace_back(idxs);
    }
    expr->ComputeShape("");
    return new plaidml_expr{expr};
#endif
#ifdef PLAIDML_MLIR
    std::vector<mlir::Value*> src_values;
    for (size_t i = 0; i < nsrcs; i++) {
      src_values.emplace_back(src_idxs[i]->value);
    }
    auto value = GlobalContext::get()->MakeContractionOp(  //
        getAggregationKind(agg_op),                        //
        getCombinationKind(combo_op),                      //
        src_values,                                        //
        sink_idxs->value,                                  //
        sink_sizes->value,                                 //
        name);
    return new plaidml_expr{value};
#endif
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
#ifdef PLAIDML_AST
    auto cion = std::dynamic_pointer_cast<ContractionExpr>(expr->expr);
    if (!cion) {
      throw std::runtime_error("add_constraint can only be specified on a contraction.");
    }
    auto constraint = std::make_shared<ConstraintExpr>(lhs->expr, rhs->expr);
    cion->constraints.emplace_back(constraint);
#endif
#ifdef PLAIDML_MLIR
    GlobalContext::get()->AddConstraint(expr->value, lhs->value, rhs->value);
#endif
  });
}

void plaidml_expr_contraction_set_no_reduce(  //
    plaidml_error* err,                       //
    plaidml_expr* expr,                       //
    bool no_reduce) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_contraction_set_no_reduce");
    if (!expr) {
      throw std::runtime_error("no_reduce can only be specified on a contraction.");
    }
#ifdef PLAIDML_AST
    auto cion = std::dynamic_pointer_cast<ContractionExpr>(expr->expr);
    if (!cion) {
      throw std::runtime_error("no_reduce can only be specified on a contraction.");
    }
    cion->no_defract = no_reduce;
#endif
#ifdef PLAIDML_MLIR
    GlobalContext::get()->SetNoReduce(expr->value, no_reduce);
#endif
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
#ifdef PLAIDML_AST
    auto cion = std::dynamic_pointer_cast<ContractionExpr>(expr->expr);
    if (!cion) {
      throw std::runtime_error("use_default can only be specified on a contraction.");
    }
    cion->use_default = use_default->expr;
#endif
#ifdef PLAIDML_MLIR
    GlobalContext::get()->SetUseDefault(expr->value, use_default->value);
#endif
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
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_gradient");
#ifdef PLAIDML_AST
    std::vector<ExprPtr> wrt_exprs(nwrts);
    for (size_t i = 0; i < nwrts; i++) {
      wrt_exprs[i] = wrts[i]->expr;
    }
    auto deriv_exprs = ComputeGradients(wrt_exprs, loss->expr);
    for (size_t i = 0; i < nwrts; i++) {
      derivs[i] = new plaidml_expr{deriv_exprs[i]};
    }
#endif
#ifdef PLAIDML_MLIR
    std::vector<mlir::Value*> wrt_values(nwrts);
    for (size_t i = 0; i < nwrts; i++) {
      wrt_values[i] = wrts[i]->value;
    }
    auto deriv_values = GlobalContext::get()->ComputeGradients(wrt_values, loss->value);
    for (size_t i = 0; i < nwrts; i++) {
      derivs[i] = new plaidml_expr{deriv_values[i]};
    }
#endif
  });
}

void plaidml_deriv_register(  //
    plaidml_error* err,       //
    const char* name,         //
    plaidml_deriv fn,         //
    void* user_ctx) {
  ffi_wrap_void(err, [&] {
    IVLOG(5, "plaidml_deriv_register> " << name);
#ifdef PLAIDML_AST
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
    DerivRegistry::Instance()->Register(name, thunk, reinterpret_cast<void*>(fn), user_ctx);
#endif
#ifdef PLAIDML_MLIR
    auto thunk = [](mlir::Value* Y,                                //
                    mlir::Value* dY,                               //
                    const llvm::SmallVector<mlir::Value*, 3>& Xs,  //
                    void* user_fn,                                 //
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
      llvm::SmallVector<mlir::Value*, 3> ret(Xs.size());
      for (size_t i = 0; i < ret.size(); i++) {
        ret[i] = dX_exprs[i]->value;
        delete dX_exprs[i];  // TODO: Do I want this delete?
      }
      return ret;
    };
    DerivRegistry::Instance()->Register(name, thunk, reinterpret_cast<void*>(fn), user_ctx);
#endif
  });
}

void plaidml_poly_expr_free(plaidml_error* err, plaidml_poly_expr* expr) {
  ffi_wrap_void(err, [&] {
#ifdef PLAIDML_AST
    IVLOG(3, "plaidml_poly_expr_free> " << expr->expr->str());
#endif
#ifdef PLAIDML_MLIR
    IVLOG(3, "plaidml_poly_expr_free> " << mlir::debugString(*expr->value));
    GlobalContext::get()->Destroy(expr->value);
#endif
    delete expr;
  });
}

plaidml_string* plaidml_poly_expr_repr(  //
    plaidml_error* err,                  //
    plaidml_poly_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_repr");
#ifdef PLAIDML_AST
    return new plaidml_string{expr->expr->str()};
#endif
#ifdef PLAIDML_MLIR
    return new plaidml_string{mlir::debugString(*expr->value)};
#endif
  });
}

plaidml_poly_expr* plaidml_poly_expr_dim(  //
    plaidml_error* err,                    //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_dim");
#ifdef PLAIDML_AST
    return new plaidml_poly_expr{std::make_shared<PolyDimExpr>(expr->expr)};
#endif
#ifdef PLAIDML_MLIR
    return new plaidml_poly_expr{expr->value};
#endif
  });
}

plaidml_poly_expr* plaidml_poly_expr_index(  //
    plaidml_error* err,                      //
    const char* name) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_index");
#ifdef PLAIDML_AST
    return new plaidml_poly_expr{std::make_shared<PolyIndex>(next_idx_id++, std::string{name})};
#endif
#ifdef PLAIDML_MLIR
    return new plaidml_poly_expr{GlobalContext::get()->MakeAffineIndexOp(name)};
#endif
  });
}

plaidml_poly_expr* plaidml_poly_expr_literal(  //
    plaidml_error* err,                        //
    int64_t value) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_literal> " << value);
#ifdef PLAIDML_AST
    return new plaidml_poly_expr{std::make_shared<PolyLiteral>(value)};
#endif
#ifdef PLAIDML_MLIR
    return new plaidml_poly_expr{GlobalContext::get()->MakeAffineConstantOp(value)};
#endif
  });
}

plaidml_poly_expr* plaidml_poly_expr_op(  //
    plaidml_error* err,                   //
    plaidml_int_op op,                    //
    size_t nargs,                         //
    plaidml_poly_expr** args) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_op> " << op);
#ifdef PLAIDML_AST
    std::vector<std::shared_ptr<PolyExpr>> vec_args(nargs);
    for (size_t i = 0; i < nargs; i++) {
      vec_args[i] = args[i]->expr;
    }
    return new plaidml_poly_expr{MakeOp(static_cast<IntOp>(op), vec_args)};
#endif
#ifdef PLAIDML_MLIR
    std::vector<mlir::Value*> values(nargs);
    for (size_t i = 0; i < nargs; i++) {
      values[i] = args[i]->value;
    }
    return new plaidml_poly_expr{MakeAffineOp(op, values)};
#endif
  });
}

void plaidml_dim_expr_free(  //
    plaidml_error* err,      //
    plaidml_dim_expr* expr) {
  ffi_wrap_void(err, [&] {
#ifdef PLAIDML_AST
    IVLOG(3, "plaidml_dim_expr_free> " << expr->expr->str());
#endif
#ifdef PLAIDML_MLIR
    IVLOG(3, "plaidml_dim_expr_free> " << mlir::debugString(*expr->value));
    GlobalContext::get()->Destroy(expr->value);
#endif
    delete expr;
  });
}

plaidml_string* plaidml_dim_expr_repr(  //
    plaidml_error* err,                 //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_repr");
#ifdef PLAIDML_AST
    return new plaidml_string{expr->expr->str()};
#endif
#ifdef PLAIDML_MLIR
    return new plaidml_string{mlir::debugString(*expr->value)};
#endif
  });
}

plaidml_dim_expr* plaidml_dim_expr_none(  //
    plaidml_error* err                    //
) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_none");
#ifdef PLAIDML_AST
    return new plaidml_dim_expr{std::make_shared<DimNoneExpr>()};
#endif
#ifdef PLAIDML_MLIR
    return new plaidml_dim_expr{GlobalContext::get()->MakeNoneOp()};
#endif
  });
}

plaidml_dim_expr* plaidml_dim_expr_int(  //
    plaidml_error* err,                  //
    int64_t value) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_int> " << value);
#ifdef PLAIDML_AST
    return new plaidml_dim_expr{std::make_shared<DimIntExpr>(value)};
#endif
#ifdef PLAIDML_MLIR
    return new plaidml_dim_expr{GlobalContext::get()->MakeAffineConstantOp(value)};
#endif
  });
}

int64_t plaidml_dim_expr_get_int(  //
    plaidml_error* err,            //
    plaidml_dim_expr* expr) {
  return ffi_wrap<int64_t>(err, 0, [&] {
    IVLOG(3, "plaidml_dim_expr_get_int");
    if (!expr) {
      throw std::runtime_error("plaidml_dim_expr_get_int can only be used on an integer value");
    }
#ifdef PLAIDML_AST
    auto int_expr = std::dynamic_pointer_cast<DimIntExpr>(expr->expr);
    if (!int_expr) {
      throw std::runtime_error("plaidml_dim_expr_get_int can only be used on an integer value");
    }
    return int_expr->value;
#endif
#ifdef PLAIDML_MLIR
    return GlobalContext::get()->GetIntegerValue(expr->value);
#endif
  });
}

plaidml_dim_expr* plaidml_dim_expr_op(  //
    plaidml_error* err,                 //
    plaidml_int_op op,                  //
    size_t nargs,                       //
    plaidml_dim_expr** args) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_op> " << op);
#ifdef PLAIDML_AST
    std::vector<DimExprPtr> vec_args(nargs);
    for (size_t i = 0; i < nargs; i++) {
      vec_args[i] = args[i]->expr;
    }
    return new plaidml_dim_expr{MakeOp(static_cast<IntOp>(op), vec_args)};
#endif
#ifdef PLAIDML_MLIR
    std::vector<mlir::Value*> values(nargs);
    for (size_t i = 0; i < nargs; i++) {
      values[i] = args[i]->value;
    }
    return new plaidml_dim_expr{MakeAffineOp(op, values)};
#endif
  });
}

void plaidml_tuple_free(  //
    plaidml_error* err,   //
    plaidml_tuple* tuple) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_tuple_free");
    for (size_t i = 0; i < tuple->nelts; i++) {
      delete tuple->elts[i];
    }
    delete[] tuple->elts;
    delete tuple;
  });
}

void plaidml_value_free(  //
    plaidml_error* err,   //
    plaidml_value* value) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_value_free");
    delete value;
  });
}

plaidml_value* plaidml_value_clone(  //
    plaidml_error* err,              //
    plaidml_value* value) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_clone");
    return new plaidml_value{*value};
  });
}

plaidml_value_kind plaidml_value_get_kind(  //
    plaidml_error* err,                     //
    plaidml_value* value) {
  return ffi_wrap<plaidml_value_kind>(err, PLAIDML_VALUE_NONE, [&] {
    IVLOG(3, "plaidml_value_get_kind");
    return std::visit(
        [](auto&& arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, plaidml_dim_expr>) {
            return PLAIDML_VALUE_DIM;
          } else if constexpr (std::is_same_v<T, plaidml_expr>) {
            return PLAIDML_VALUE_EXPR;
          } else if constexpr (std::is_same_v<T, double>) {
            return PLAIDML_VALUE_FLOAT;
          } else if constexpr (std::is_same_v<T, int64_t>) {
            return PLAIDML_VALUE_INT;
          } else if constexpr (std::is_same_v<T, std::string>) {
            return PLAIDML_VALUE_STR;
          } else if constexpr (std::is_same_v<T, Tuple>) {
            return PLAIDML_VALUE_TUPLE;
          } else {
            return PLAIDML_VALUE_NONE;
          }
        },
        value->variant);
  });
}

plaidml_value* plaidml_value_none(  //
    plaidml_error* err              //
) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_none");
    return new plaidml_value{};
  });
}

plaidml_value* plaidml_value_int(  //
    plaidml_error* err,            //
    int64_t value) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_int> " << value);
    return new plaidml_value{value};
  });
}

plaidml_value* plaidml_value_dim(  //
    plaidml_error* err,            //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_dim");
    if (!expr) {
      throw std::runtime_error("plaidml_value_dim requires non-null expr");
    }
    return new plaidml_value{*expr};
  });
}

plaidml_value* plaidml_value_expr(  //
    plaidml_error* err,             //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_expr> " << expr);
    if (!expr) {
      throw std::runtime_error("plaidml_value_expr requires non-null expr");
    }
    return new plaidml_value{*expr};
  });
}

plaidml_value* plaidml_value_float(  //
    plaidml_error* err,              //
    double value) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_float");
    return new plaidml_value{value};
  });
}

plaidml_value* plaidml_value_str(  //
    plaidml_error* err,            //
    const char* value) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_str> " << value);
    return new plaidml_value{value};
  });
}

plaidml_value* plaidml_value_tuple(  //
    plaidml_error* err,              //
    size_t nelts,                    //
    plaidml_value** elts) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_tuple");
    Tuple tuple(nelts);
    for (size_t i = 0; i < nelts; i++) {
      tuple[i] = std::make_shared<VariantHolder>(elts[i]->variant);
    }
    return new plaidml_value{tuple};
  });
}

plaidml_dim_expr* plaidml_value_dim_get(  //
    plaidml_error* err,                   //
    plaidml_value* value) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_dim_get");
    return new plaidml_dim_expr{std::get<plaidml_dim_expr>(value->variant)};
  });
}

plaidml_expr* plaidml_value_expr_get(  //
    plaidml_error* err,                //
    plaidml_value* value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_expr_get");
    auto expr = std::get<plaidml_expr>(value->variant);
    return new plaidml_expr{expr};
  });
}

double plaidml_value_float_get(  //
    plaidml_error* err,          //
    plaidml_value* value) {
  return ffi_wrap<double>(err, 0, [&] {
    IVLOG(3, "plaidml_value_float_get");
    return std::get<double>(value->variant);
  });
}

int64_t plaidml_value_int_get(  //
    plaidml_error* err,         //
    plaidml_value* value) {
  return ffi_wrap<int64_t>(err, 0, [&] {
    IVLOG(3, "plaidml_value_int_get");
    return std::get<int64_t>(value->variant);
  });
}

plaidml_tuple* plaidml_value_tuple_get(  //
    plaidml_error* err,                  //
    plaidml_value* value) {
  return ffi_wrap<plaidml_tuple*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_tuple_get");
    auto tuple = std::get<Tuple>(value->variant);
    auto nelts = tuple.size();
    auto elts = new plaidml_value*[nelts];
    for (size_t i = 0; i < nelts; i++) {
      elts[i] = new plaidml_value{tuple[i]->inner};
    }
    return new plaidml_tuple{nelts, elts};
  });
}

plaidml_string* plaidml_value_str_get(  //
    plaidml_error* err,                 //
    plaidml_value* value) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_str_get");
    return new plaidml_string{std::get<std::string>(value->variant)};
  });
}

plaidml_string* plaidml_value_repr(  //
    plaidml_error* err,              //
    plaidml_value* value) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_repr");
    auto str = std::visit(
        [](auto&& arg) -> std::string {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, plaidml_dim_expr>) {
#ifdef PLAIDML_AST
            return arg.expr->str();
#endif
#ifdef PLAIDML_MLIR
            return mlir::debugString(*arg.value);
#endif
          } else if constexpr (std::is_same_v<T, plaidml_expr>) {
#ifdef PLAIDML_AST
            return arg.expr->str();
#endif
#ifdef PLAIDML_MLIR
            return mlir::debugString(*arg.value);
#endif
          } else if constexpr (std::is_same_v<T, double>) {
            return std::to_string(arg);
          } else if constexpr (std::is_same_v<T, int64_t>) {
            return std::to_string(arg);
          } else if constexpr (std::is_same_v<T, std::string>) {
            return llvm::formatv("\"{0}\"", arg).str();
          } else if constexpr (std::is_same_v<T, Tuple>) {
            return "[]";
          } else {
            return "None";
          }
        },
        value->variant);
    return new plaidml_string{str};
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
    size_t nupdates,                        //
    plaidml_expr** src_updates,             //
    plaidml_expr** dst_updates,             //
    plaidml_program_args** raw_args) {
  return ffi_wrap<plaidml_program*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_program_evaluate");
#ifdef PLAIDML_AST
    ProgramMutations mutations;
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
    auto nargs = ret->eval.args.size();
    auto args = new plaidml_program_arg[nargs];
    for (unsigned i = 0; i < nargs; i++) {
      const auto& arg = ret->eval.args[i];
      args[i].is_input = arg.is_input;
      args[i].tensor = new plaidml_expr{arg.expr};
      args[i].shape = new plaidml_logical_shape{arg.expr->shape};
      auto param_expr = std::dynamic_pointer_cast<ParamExpr>(arg.expr);
      if (param_expr && param_expr->buffer) {
        args[i].buffer = new plaidml_buffer{param_expr->buffer};
      } else {
        args[i].buffer = nullptr;
      }
    }
    *raw_args = new plaidml_program_args{nargs, args};
    return ret;
#endif
#ifdef PLAIDML_MLIR
    IVLOG(5, "  plaidml_program_evaluate>> noutputs: " << noutputs << ", nupdates: " << nupdates);
    ProgramMutations mutations;
    for (size_t i = 0; i < noutputs; i++) {
      if (!raw_outputs[i]) {
        throw std::runtime_error("Undefined output in plaidml_program_evaluate");
      }
      mutations.outputs.emplace_back(raw_outputs[i]->value);
    }
    for (size_t i = 0; i < nupdates; i++) {
      if (!src_updates[i]) {
        throw std::runtime_error("Undefined update src in plaidml_program_evaluate");
      }
      if (!dst_updates[i]) {
        throw std::runtime_error("Undefined update dst in plaidml_program_evaluate");
      }
      mutations.updates.emplace(ProgramUpdate{src_updates[i]->value, dst_updates[i]->value});
    }
    auto ret = new plaidml_program{GlobalContext::get()->MakeProgram(name, mutations)};
    assert(noutputs <= ret->program->outputs.size());
    auto nargs = ret->program->arguments.size();
    auto args = new plaidml_program_arg[nargs];
    for (unsigned i = 0; i < nargs; i++) {
      args[i].is_input = ret->program->arguments[i].isInput;
      args[i].tensor = new plaidml_expr{ret->program->arguments[i].value};
      args[i].shape = new plaidml_logical_shape{ret->program->arguments[i].shape};
      if (ret->program->arguments[i].buffer) {
        args[i].buffer = new plaidml_buffer{ret->program->arguments[i].buffer};
      } else {
        args[i].buffer = nullptr;
      }
    }
    *raw_args = new plaidml_program_args{nargs, args};
    return ret;
#endif
  });
}

plaidml_string* plaidml_program_repr(  //
    plaidml_error* err,                //
    plaidml_program* program) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_program_repr");
#ifdef PLAIDML_AST
    return new plaidml_string{to_string(program->eval.runinfo.program)};
#endif
#ifdef PLAIDML_MLIR
    auto module = *program->program->module;
    return new plaidml_string{mlir::debugString(module)};
#endif
  });
}

void plaidml_program_args_free(  //
    plaidml_error* err,          //
    plaidml_program_args* args) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_program_args_free");
    for (unsigned i = 0; i < args->nargs; i++) {
      delete args->args[i].shape;
      delete args->args[i].tensor;
      delete args->args[i].buffer;
    }
    delete[] args->args;
    delete args;
  });
}

}  // extern "C"
