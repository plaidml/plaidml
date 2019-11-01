// Copyright 2019 Intel Corporation.

#include "plaidml2/edsl/ffi.h"

#include <algorithm>
#include <vector>

#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"

#include "base/util/logging.h"
#include "plaidml2/core/internal.h"
#include "plaidml2/edsl/derivs.h"
#include "pmlc/dialect/eltwise/types.h"
#include "pmlc/dialect/tile/ops.h"
#include "pmlc/util/enums.h"

using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using plaidml::core::GlobalContext;
using pmlc::dialect::eltwise::ScalarType;
using pmlc::dialect::tile::TileBuilder;
using pmlc::util::AggregationKind;
using pmlc::util::CombinationKind;
using vertexai::tile::DataType;

namespace {

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

}  // namespace

extern "C" {

struct plaidml_logical_shape {
  mlir::RankedTensorType type;
};

struct plaidml_dim_expr {
  mlir::Value* value = nullptr;
};

struct plaidml_poly_expr {
  mlir::Value* value = nullptr;
};

void plaidml_edsl_init(  //
    plaidml_error* err) {
  static std::once_flag is_initialized;
  ffi_wrap_void(err, [&] {
    std::call_once(is_initialized, []() {
      IVLOG(1, "plaidml_edsl_init");
      // plaidml::edsl::deriv::RegisterDerivs();
    });
  });
}

plaidml_logical_shape* plaidml_logical_shape_alloc(  //
    plaidml_error* err,                              //
    plaidml_datatype dtype,                          //
    size_t ndims,                                    //
    const int64_t* dims) {
  return ffi_wrap<plaidml_logical_shape*>(err, nullptr, [&] {
    llvm::SmallVector<int64_t, 6> dimsVec;
    for (size_t i = 0; i < ndims; i++) {
      dimsVec.emplace_back(dims[i]);
    }
    auto ret = new plaidml_logical_shape;
    ret->type = GlobalContext::get()->MakeRankedTensorType(static_cast<DataType>(dtype), dimsVec);
    return ret;
  });
}

plaidml_string* plaidml_logical_shape_repr(  //
    plaidml_error* err,                      //
    plaidml_logical_shape* shape) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {  //
    return new plaidml_string{mlir::debugString(shape->type)};
  });
}

size_t plaidml_logical_shape_get_ndims(  //
    plaidml_error* err,                  //
    plaidml_logical_shape* shape) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return shape->type.getRank();
  });
}

plaidml_datatype plaidml_logical_shape_get_dtype(  //
    plaidml_error* err,                            //
    plaidml_logical_shape* shape) {
  return ffi_wrap<plaidml_datatype>(err, PLAIDML_DATA_INVALID, [&] {
    auto elementType = shape->type.getElementType();
    auto scalarType = elementType.dyn_cast<ScalarType>();
    if (!scalarType) {
      throw std::runtime_error("Expected scalar type");
    }
    return static_cast<plaidml_datatype>(scalarType.type());
  });
}

int64_t plaidml_logical_shape_get_dim_int(  //
    plaidml_error* err,                     //
    plaidml_logical_shape* shape,           //
    size_t dim) {
  return ffi_wrap<int64_t>(err, 0, [&] {
    const auto& dims = shape->type.getShape();
    if (dims.size() < dim) {
      throw std::range_error("Index out of range");
    }
    auto ret = dims[dim];
    if (ret < 0) {
      return static_cast<int64_t>(0);
    }
    return ret;
  });
}

// plaidml_dim_expr* plaidml_logical_shape_get_dim_expr(  //
//     plaidml_error* err,                                //
//     plaidml_logical_shape* shape,                      //
//     size_t dim) {
//   return ffi_wrap<plaidml_dim_expr*>(err, 0, [&] {  //
//     return new plaidml_dim_expr{shape->shape.dims.at(dim).expr};
//   });
// }

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
    return new plaidml_shape{GlobalContext::get()->IntoTensorType(shape->type)};
  });
}

void plaidml_expr_free(  //
    plaidml_error* err,  //
    plaidml_expr* expr) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_free> " << expr->value);
    GlobalContext::get()->Destroy(expr->value);
    delete expr;
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
    return new plaidml_logical_shape{GlobalContext::get()->ComputeShape(expr->value)};
  });
}

void plaidml_expr_bind_shape(  //
    plaidml_error* err,        //
    plaidml_expr* expr,        //
    plaidml_logical_shape* shape) {
  return ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_bind_shape");
    GlobalContext::get()->BindShape(expr->value, shape->type);
  });
}

void plaidml_expr_bind_dims(  //
    plaidml_error* err,       //
    plaidml_expr* expr,       //
    size_t ndims,             //
    plaidml_dim_expr** dims) {
  return ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_bind_dims> " << mlir::debugString(*expr->value));
    llvm::SmallVector<mlir::Value**, 6> into;
    for (size_t i = 0; i < ndims; i++) {
      IVLOG(3, "bind_dims> i: " << i << ", from: " << expr->value << ", into: " << dims[i]->value);
      into.emplace_back(&dims[i]->value);
    }
    GlobalContext::get()->BindTensorDims(expr->value, into);
  });
}

plaidml_string* plaidml_expr_repr(  //
    plaidml_error* err,             //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_repr");
    return new plaidml_string{mlir::debugString(*expr->value)};
  });
}

plaidml_expr* plaidml_expr_dim(  //
    plaidml_error* err,          //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_dim");
    // TODO: clone?
    return new plaidml_expr{expr->value};
  });
}

plaidml_expr* plaidml_expr_placeholder(  //
    plaidml_error* err,                  //
    plaidml_logical_shape* shape,        //
    plaidml_buffer* buffer,              //
    const char* name) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_placeholder");
    return new plaidml_expr{
        GlobalContext::get()->MakePlaceholderOp(shape->type, buffer ? buffer->buffer : nullptr, name)};
  });
}

void plaidml_expr_param_reset(  //
    plaidml_error* err,         //
    plaidml_expr* expr,         //
    plaidml_buffer* buffer) {
  return ffi_wrap_void(err, [&] {
    throw std::runtime_error("NYI: plaidml_expr_param_reset");
    // auto param_expr = std::dynamic_pointer_cast<ParamExpr>(expr->expr);
    // if (param_expr) {
    //   param_expr->buffer = buffer->buffer;
    // } else {
    //   throw std::runtime_error("ParamExpr value reset requested for non-ParamExpr");
    // }
  });
}

plaidml_expr* plaidml_expr_clone(  //
    plaidml_error* err,            //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_clone> " << mlir::debugString(*expr->value));
    // TODO(MLIR): deal with clone of expr->value
    return new plaidml_expr{expr->value};
  });
}

plaidml_dim_expr* plaidml_expr_get_dim(  //
    plaidml_error* err,                  //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_get_dim> " << mlir::debugString(*expr->value));
    // auto dim_expr = std::dynamic_pointer_cast<DimExprExpr>(expr->expr);
    // if (!dim_expr) {
    //   throw std::runtime_error("plaidml_expr_get_dim can only be used on a DimExprExpr");
    // }
    // TODO(MLIR): deal with clone of expr->value
    return new plaidml_dim_expr{expr->value};
  });
}

plaidml_expr_kind plaidml_expr_get_kind(  //
    plaidml_error* err,                   //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_expr_kind>(err, PLAIDML_EXPR_NONE, [&] {
    IVLOG(3, "plaidml_expr_get_kind");
    auto op = expr->value->getDefiningOp();
    if (auto constOp = llvm::dyn_cast_or_null<pmlc::dialect::eltwise::ScalarConstantOp>(op)) {
      auto attr = constOp.getValue();
      if (attr.isa<mlir::IntegerAttr>()) {
        return PLAIDML_EXPR_INT;
      }
      if (attr.isa<mlir::FloatAttr>()) {
        return PLAIDML_EXPR_FLOAT;
      }
    }
    auto type = expr->value->getType();
    if (type.isa<mlir::NoneType>()) {
      return PLAIDML_EXPR_NONE;
    }
    if (type.isa<pmlc::dialect::tile::StringType>()) {
      return PLAIDML_EXPR_STR;
    }
    if (type.isa<mlir::TupleType>()) {
      return PLAIDML_EXPR_TUPLE;
    }
    if (type.isa<mlir::IndexType>()) {
      return PLAIDML_EXPR_DIM;
    }
    if (type.isa<mlir::RankedTensorType>()) {
      return PLAIDML_EXPR_TENSOR;
    }
    throw std::runtime_error("Unknown expression kind");
  });
}

plaidml_expr* plaidml_expr_none(  //
    plaidml_error* err            //
) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_none");
    return new plaidml_expr{GlobalContext::get()->MakeNoneOp()};
  });
}

plaidml_expr* plaidml_expr_tuple(  //
    plaidml_error* err,            //
    size_t nargs,                  //
    plaidml_expr** args) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_tuple");
    std::vector<mlir::Value*> tuple(nargs);
    for (size_t i = 0; i < nargs; i++) {
      tuple[i] = args[i]->value;
    }
    return new plaidml_expr{GlobalContext::get()->MakeTupleOp(tuple)};
  });
}

size_t plaidml_expr_tuple_get_count(  //
    plaidml_error* err,               //
    plaidml_expr* expr) {
  return ffi_wrap<size_t>(err, 0, [&] {
    IVLOG(3, "plaidml_expr_tuple_get_count");
    auto elts = GlobalContext::get()->GetTupleElements(expr->value);
    return elts.size();
  });
}

void plaidml_expr_tuple_get_exprs(  //
    plaidml_error* err,             //
    plaidml_expr* expr,             //
    size_t nexprs,                  //
    plaidml_expr** exprs) {
  return ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_tuple_get_exprs> nexprs: " << nexprs);
    auto elts = GlobalContext::get()->GetTupleElements(expr->value);
    for (size_t i = 0; i < std::min(nexprs, elts.size()); i++) {
      exprs[i] = new plaidml_expr{elts[i]};
    }
  });
}

plaidml_expr* plaidml_expr_str(  //
    plaidml_error* err,          //
    const char* value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_str> " << value);
    return new plaidml_expr{GlobalContext::get()->MakeStringOp(value)};
  });
}

plaidml_string* plaidml_expr_str_get_value(  //
    plaidml_error* err,                      //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_str_get_value");
    if (!expr) {
      throw std::runtime_error("plaidml_expr_str_get_value can only be used on an StringExpr");
    }
    auto str = GlobalContext::get()->GetStringValue(expr->value);
    return new plaidml_string{str.str()};
  });
}

plaidml_expr* plaidml_expr_int(  //
    plaidml_error* err,          //
    int64_t value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_int> " << value);
    return new plaidml_expr{GlobalContext::get()->MakeScalarConstantOp(value)};
  });
}

int64_t plaidml_expr_int_get_value(  //
    plaidml_error* err,              //
    plaidml_expr* expr) {
  return ffi_wrap<int64_t>(err, 0, [&] {
    IVLOG(3, "plaidml_expr_int_get_value");
    if (!expr) {
      throw std::runtime_error("plaidml_expr_int_get_value can only be used on an IntConst");
    }
    return GlobalContext::get()->GetIntegerValue(expr->value);
  });
}

plaidml_expr* plaidml_expr_float(  //
    plaidml_error* err,            //
    double value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_float");
    return new plaidml_expr{GlobalContext::get()->MakeScalarConstantOp(value)};
  });
}

double plaidml_expr_float_get_value(  //
    plaidml_error* err,               //
    plaidml_expr* expr) {
  return ffi_wrap<double>(err, 0, [&] {
    IVLOG(3, "plaidml_expr_float_get_value");
    return GlobalContext::get()->GetFloatValue(expr->value);
  });
}

plaidml_expr* plaidml_expr_cast(  //
    plaidml_error* err,           //
    plaidml_expr* tensor,         //
    plaidml_datatype dtype) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_cast");
    return new plaidml_expr{GlobalContext::get()->MakeCastOp(tensor->value, static_cast<DataType>(dtype))};
  });
}

plaidml_expr* plaidml_expr_call(  //
    plaidml_error* err,           //
    const char* fn,               //
    size_t nargs,                 //
    plaidml_expr** args) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_call");
    std::vector<mlir::Value*> values(nargs);
    for (size_t i = 0; i < nargs; i++) {
      values[i] = args[i]->value;
    }
    return new plaidml_expr{GlobalContext::get()->MakePrimitiveOp(fn, values)};
  });
}

plaidml_expr* plaidml_expr_grad_override(  //
    plaidml_error* err,                    //
    plaidml_deriv fn,                      //
    void* user_ctx,                        //
    size_t nins,                           //
    plaidml_expr** ins,                    //
    plaidml_expr* out) {
  // auto thunk = [](const ExprPtr& Y,                //
  //                 const ExprPtr& dY,               //
  //                 const std::vector<ExprPtr>& Xs,  //
  //                 void* user_fn,                   //
  //                 void* user_ctx) {
  //   IVLOG(6, "plaidml_grad_override> thunk");
  //   plaidml_deriv fn = reinterpret_cast<plaidml_deriv>(user_fn);
  //   std::vector<plaidml_expr*> X_exprs(Xs.size());
  //   std::vector<plaidml_expr*> dX_exprs(Xs.size());
  //   // TODO: Construct the plaidml_exprs here w/ MLIR values too?
  //   auto Y_expr = new plaidml_expr{Y};
  //   auto dY_expr = new plaidml_expr{dY};
  //   for (size_t i = 0; i < X_exprs.size(); i++) {
  //     X_exprs[i] = new plaidml_expr{Xs[i]};
  //   }
  //   fn(user_ctx, Y_expr, dY_expr, X_exprs.size(), X_exprs.data(), dX_exprs.data());
  //   std::vector<ExprPtr> ret(Xs.size());
  //   for (size_t i = 0; i < ret.size(); i++) {
  //     ret[i] = dX_exprs[i]->expr;
  //     delete dX_exprs[i];
  //   }
  //   return ret;
  // };
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(5, "plaidml_grad_override");
    throw std::runtime_error("NYI: plaidml_grad_override");
    return nullptr;
    // auto deriv_entry = std::make_shared<ExprDerivEntry>();
    // deriv_entry->fn = thunk;
    // deriv_entry->user_fn = reinterpret_cast<void*>(fn);
    // deriv_entry->user_ctx = user_ctx;
    // std::vector<ExprPtr> in_exprs(nins);
    // ExprPtr out_expr;
    // mlir::Value* out_value;
    // for (size_t i = 0; i < nins; i++) {
    //   if (!ins[i]) {
    //     throw std::runtime_error("Undefined input tensor in gradient override");
    //   }
    //   if (ins[i]->expr) {
    //     in_exprs[i] = ins[i]->expr;
    //   } else {
    //     throw std::runtime_error("Currently exprs are required in plaidml_grad_override");  // TODO: MLIR
    //   }
    // }
    // if (!out) {
    //   throw std::runtime_error("Undefined output tensor in gradient override");
    // }
    // if (out->expr) {
    //   out_expr = out->expr;
    // } else {
    //   throw std::runtime_error("Currently out expr is required in plaidml_expr_grad_override");  // TODO: MLIR
    // }
    // if (out->value && use_mlir()) {
    //   out_value = out->value;
    // } else {
    //   out_value = nullptr;
    // }
    // ExprPtr expr = MakeGradOverride(deriv_entry, in_exprs, out_expr);
    // mlir::Value* value = use_mlir() ? GlobalContext::get()->MakePrimitiveOp("ident", out_value) : nullptr;
    // IVLOG(6, "The expr from plaidml_expr_grad_override has shape " << expr->shape.str());
    // return new plaidml_expr{expr, value};
  });
}

plaidml_expr* plaidml_expr_index_map(  //
    plaidml_error* err,                //
    plaidml_expr* ref,                 //
    size_t ndims,                      //
    plaidml_poly_expr** raw_idxs) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_index_map");
    std::vector<mlir::Value*> idx_values(ndims);
    for (size_t i = 0; i < ndims; i++) {
      idx_values[i] = raw_idxs[i]->value;
    }
    if (ref) {
      return new plaidml_expr{GlobalContext::get()->MakeAffineSourceIndexMapOp(ref->value, idx_values)};
    }
    return new plaidml_expr{GlobalContext::get()->MakeAffineSinkIndexMapOp(idx_values)};
  });
}

plaidml_expr* plaidml_expr_size_map(  //
    plaidml_error* err,               //
    size_t ndims,                     //
    plaidml_dim_expr** raw_dims) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_size_map");
    std::vector<mlir::Value*> dim_values;
    for (size_t i = 0; i < ndims; i++) {
      dim_values.emplace_back(raw_dims[i]->value);
    }
    return new plaidml_expr{GlobalContext::get()->MakeAffineSizeMapOp(dim_values)};
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
    llvm::SmallVector<mlir::Value*, 3> values;
    for (size_t i = 0; i < nsrcs; i++) {
      values.emplace_back(src_idxs[i]->value);
    }
    auto agg = getAggregationKind(agg_op);
    auto combo = getCombinationKind(combo_op);
    auto value = GlobalContext::get()->MakeContractionOp(agg, combo, values, sink_idxs->value, sink_sizes->value, name);
    return new plaidml_expr{value};
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
    GlobalContext::get()->AddConstraint(expr->value, lhs->value, rhs->value);
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
    throw std::runtime_error("NYI: plaidml_expr_contraction_set_no_defract");
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
    GlobalContext::get()->SetUseDefault(expr->value, use_default->value);
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
    std::vector<mlir::Value*> wrt_values(nwrts);
    for (size_t i = 0; i < nwrts; i++) {
      wrt_values[i] = wrts[i]->value;
    }
    auto deriv_values = GlobalContext::get()->ComputeGradients(wrt_values, loss->value);
    for (size_t i = 0; i < nwrts; i++) {
      derivs[i] = new plaidml_expr{deriv_values[i]};
    }
  });
}

void plaidml_deriv_register(  //
    plaidml_error* err,       //
    const char* name,         //
    plaidml_deriv fn,         //
    void* user_ctx) {
  // TODO(MLIR)
  // auto thunk = [](const ExprPtr& Y,                //
  //                 const ExprPtr& dY,               //
  //                 const std::vector<ExprPtr>& Xs,  //
  //                 void* user_fn,                   //
  //                 void* user_ctx) {
  //   IVLOG(6, "plaidml_deriv_register> thunk");
  //   plaidml_deriv fn = reinterpret_cast<plaidml_deriv>(user_fn);
  //   std::vector<plaidml_expr*> X_exprs(Xs.size());
  //   std::vector<plaidml_expr*> dX_exprs(Xs.size());
  //   auto Y_expr = new plaidml_expr{Y};
  //   auto dY_expr = new plaidml_expr{dY};
  //   for (size_t i = 0; i < X_exprs.size(); i++) {
  //     X_exprs[i] = new plaidml_expr{Xs[i]};
  //   }
  //   fn(user_ctx, Y_expr, dY_expr, X_exprs.size(), X_exprs.data(), dX_exprs.data());
  //   std::vector<ExprPtr> ret(Xs.size());
  //   for (size_t i = 0; i < ret.size(); i++) {
  //     ret[i] = dX_exprs[i]->expr;
  //     delete dX_exprs[i];
  //   }
  //   return ret;
  // };
  ffi_wrap_void(err, [&] {
    IVLOG(5, "plaidml_deriv_register> " << name);
    // DerivRegistry::Instance()->Register(name, thunk, reinterpret_cast<void*>(fn), user_ctx);
    throw std::runtime_error("NYI: plaidml_deriv_register");
  });
}

void plaidml_poly_expr_free(plaidml_error* err, plaidml_poly_expr* expr) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_poly_expr_free> " << mlir::debugString(*expr->value));
    GlobalContext::get()->Destroy(expr->value);
    delete expr;
  });
}

plaidml_string* plaidml_poly_expr_repr(  //
    plaidml_error* err,                  //
    plaidml_poly_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {  //
    IVLOG(3, "plaidml_poly_expr_repr");
    return new plaidml_string{mlir::debugString(*expr->value)};
  });
}

PLAIDML_EDSL_API plaidml_poly_expr* plaidml_poly_expr_dim(  //
    plaidml_error* err,                                     //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {  //
    IVLOG(3, "plaidml_poly_expr_dim");
    return new plaidml_poly_expr{expr->value};
  });
}

plaidml_poly_expr* plaidml_poly_expr_index(  //
    plaidml_error* err,                      //
    const char* name) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_index");
    return new plaidml_poly_expr{GlobalContext::get()->MakeAffineIndexOp(name)};
  });
}

plaidml_poly_expr* plaidml_poly_expr_literal(  //
    plaidml_error* err,                        //
    int64_t value) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_literal> " << value);
    return new plaidml_poly_expr{GlobalContext::get()->MakeAffineConstantOp(value)};
  });
}

plaidml_poly_expr* plaidml_poly_expr_op(  //
    plaidml_error* err,                   //
    plaidml_int_op op,                    //
    size_t nargs,                         //
    plaidml_poly_expr** args) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_op> " << op);
    std::vector<mlir::Value*> values(nargs);
    for (size_t i = 0; i < nargs; i++) {
      values[i] = args[i]->value;
      IVLOG(3, "  arg: " << mlir::debugString(*values[i]));
    }
    return new plaidml_poly_expr{MakeAffineOp(op, values)};
  });
}

void plaidml_dim_expr_free(  //
    plaidml_error* err,      //
    plaidml_dim_expr* expr) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_dim_expr_free> " << mlir::debugString(*expr->value));
    GlobalContext::get()->Destroy(expr->value);
    delete expr;
  });
}

plaidml_string* plaidml_dim_expr_repr(  //
    plaidml_error* err,                 //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {  //
    IVLOG(3, "plaidml_dim_expr_repr");
    return new plaidml_string{mlir::debugString(*expr->value)};
  });
}

plaidml_dim_expr* plaidml_dim_expr_none(  //
    plaidml_error* err                    //
) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {  //
    IVLOG(3, "plaidml_dim_expr_none");
    return new plaidml_dim_expr{GlobalContext::get()->MakeNoneOp()};
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
    return new plaidml_dim_expr{GlobalContext::get()->MakeAffineConstantOp(value)};
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
    return GlobalContext::get()->GetIntegerValue(expr->value);
  });
}

plaidml_dim_expr* plaidml_dim_expr_op(  //
    plaidml_error* err,                 //
    plaidml_int_op op,                  //
    size_t nargs,                       //
    plaidml_dim_expr** args) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_op> " << op);
    std::vector<mlir::Value*> values(nargs);
    for (size_t i = 0; i < nargs; i++) {
      values[i] = args[i]->value;
      IVLOG(3, "  arg: " << mlir::debugString(*values[i]));
    }
    return new plaidml_dim_expr{MakeAffineOp(op, values)};
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
    std::vector<mlir::Value*> outputs(noutputs);
    for (size_t i = 0; i < noutputs; i++) {
      if (!raw_outputs[i]) {
        throw std::runtime_error("Undefined output in plaidml_program_evaluate");
      }
      outputs[i] = raw_outputs[i]->value;
      if (!outputs[i]) {
        IVLOG(5, "Found a missing output value: (index " << i << ")");
      }
    }

    // TODO(MLIR): updates
    // std::vector<ProgramUpdate> updates(nupdates);
    // for (size_t i = 0; i < nupdates; i++) {
    //   if (!src_updates[i]) {
    //     throw std::runtime_error("Undefined update src in plaidml_program_evaluate");
    //   }
    //   if (!dst_updates[i]) {
    //     throw std::runtime_error("Undefined update dst in plaidml_program_evaluate");
    //   }
    // }

    std::vector<mlir::Value*> new_values(noutputs);
    auto program = GlobalContext::get()->MakeProgram(name, outputs, new_values);
    for (size_t i = 0; i < noutputs; i++) {
      new_outputs[i] = new plaidml_expr{new_values[i]};
    }
    // if (noutputs != ret->eval.outputs.size()) {
    //   throw std::runtime_error("Internal error: noutputs != ret->eval.outputs.size()");
    // }
    return new plaidml_program{program};
  });
}

plaidml_string* plaidml_program_repr(  //
    plaidml_error* err,                //
    plaidml_program* program) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_program_repr");
    auto module = *program->program->module;
    return new plaidml_string{mlir::debugString(module)};
  });
}

// const void* plaidml_program_runinfo(  //
//     plaidml_error* err,               //
//     plaidml_program* program) {
//   // TODO(MLIR)
//   return ffi_wrap<const void*>(err, nullptr, [&] {
//     IVLOG(3, "plaidml_program_runinfo");
//     throw std::runtime_error("NYI: plaidml_program_runinfo");
//     return nullptr;
//     // return &program->eval.runinfo;
//   });
// }

}  // extern "C"
