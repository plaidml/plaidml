// Copyright 2019 Intel Corporation.

#include "plaidml/edsl/ffi.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "llvm/Support/FormatVariadic.h"

#include "plaidml/core/internal.h"
#include "plaidml/edsl/derivs.h"
#include "pmlc/util/logging.h"

#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/dialect/tile/gradient.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/util/all_dialects.h"
#include "pmlc/util/all_passes.h"
#include "pmlc/util/enums.h"
#include "pmlc/util/env.h"

using plaidml::core::convertFromDataType;
using plaidml::core::convertIntoDataType;
using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using plaidml::core::GlobalContext;
using pmlc::dialect::tile::DerivRegistry;
using pmlc::dialect::tile::ProgramMutations;
using pmlc::dialect::tile::ProgramUpdate;
using pmlc::dialect::tile::TileBuilder;
using pmlc::util::AggregationKind;
using pmlc::util::CombinationKind;

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

mlir::Value MakePolyOp(plaidml_int_op op, const std::vector<mlir::Value> operands) {
  auto builder = GlobalContext::get();
  switch (op) {
    case PLAIDML_INT_OP_ADD:
      return builder->MakePolyAddOp(operands);
    case PLAIDML_INT_OP_DIV:
      return builder->MakePolyDivOp(operands);
    case PLAIDML_INT_OP_MUL:
      return builder->MakePolyMulOp(operands);
    case PLAIDML_INT_OP_NEG:
      return builder->MakePolyNegOp(operands);
    case PLAIDML_INT_OP_SUB:
      return builder->MakePolySubOp(operands);
    case PLAIDML_INT_OP_MAX:
      return builder->MakePolyMaxOp(operands);
    case PLAIDML_INT_OP_MIN:
      return builder->MakePolyMinOp(operands);
  }
  throw std::runtime_error("Unknown polynomial op");
}

}  // namespace

extern "C" {

struct plaidml_logical_shape {
  mlir::RankedTensorType type;
};

struct plaidml_poly_expr {
  mlir::Value value;
};

void plaidml_edsl_init(  //
    plaidml_error* err) {
  static std::once_flag is_initialized;
  ffi_wrap_void(err, [&] {
    std::call_once(is_initialized, []() {
      IVLOG(1, "plaidml_edsl_init");
      registerAllDialects();
      registerAllPasses();
      plaidml::edsl::RegisterDerivs();
    });
  });
}

plaidml_logical_shape* plaidml_logical_shape_alloc(  //
    plaidml_error* err,                              //
    plaidml_datatype dtype,                          //
    size_t rank,                                     //
    const int64_t* dims) {
  return ffi_wrap<plaidml_logical_shape*>(err, nullptr, [&] {
    llvm::SmallVector<int64_t, 6> dimsVec;
    for (size_t i = 0; i < rank; i++) {
      dimsVec.emplace_back(dims[i]);
    }
    auto ret = new plaidml_logical_shape;
    auto ctx = GlobalContext::get();
    auto elementType = convertFromDataType(dtype, ctx->getContext());
    ret->type = ctx->MakeRankedTensorType(elementType, dimsVec);
    return ret;
  });
}

plaidml_logical_shape* plaidml_logical_shape_clone(  //
    plaidml_error* err,                              //
    plaidml_logical_shape* shape) {
  return ffi_wrap<plaidml_logical_shape*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_logical_shape");
    return new plaidml_logical_shape{shape->type};
  });
}

plaidml_string* plaidml_logical_shape_repr(  //
    plaidml_error* err,                      //
    plaidml_logical_shape* shape) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] { return new plaidml_string{mlir::debugString(shape->type)}; });
}

size_t plaidml_logical_shape_get_rank(  //
    plaidml_error* err,                 //
    plaidml_logical_shape* shape) {
  return ffi_wrap<size_t>(err, 0, [&] { return shape->type.getRank(); });
}

plaidml_datatype plaidml_logical_shape_get_dtype(  //
    plaidml_error* err,                            //
    plaidml_logical_shape* shape) {
  return ffi_wrap<plaidml_datatype>(err, PLAIDML_DATA_INVALID, [&] {
    auto elementType = shape->type.getElementType();
    return convertIntoDataType(elementType);
  });
}

plaidml_integers* plaidml_logical_shape_get_sizes(  //
    plaidml_error* err,                             //
    plaidml_logical_shape* shape) {
  return ffi_wrap<plaidml_integers*>(err, 0, [&] {
    const auto& sizes = shape->type.getShape();
    auto ret = new plaidml_integers{sizes.size(), new int64_t[sizes.size()]};
    for (unsigned i = 0; i < sizes.size(); i++) {
      if (sizes[i] < 0) {
        ret->elts[i] = 0;
      } else {
        ret->elts[i] = sizes[i];
      }
    }
    return ret;
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
    return new plaidml_shape{GlobalContext::get()->IntoMemRefType(shape->type)};
  });
}

void plaidml_expr_free(  //
    plaidml_error* err,  //
    plaidml_expr* expr) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_free> " << mlir::debugString(expr->value));
    GlobalContext::get()->Destroy(expr->value);
    delete expr;
  });
}

void* plaidml_expr_ptr(  //
    plaidml_error* err,  //
    plaidml_expr* expr) {
  return ffi_wrap<void*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_ptr");
    return expr->value.getAsOpaquePointer();
  });
}

plaidml_datatype plaidml_expr_get_dtype(  //
    plaidml_error* err,                   //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_datatype>(err, PLAIDML_DATA_INVALID, [&] {
    IVLOG(3, "plaidml_expr_get_dtype");
    auto tensorType = expr->value.getType().dyn_cast<mlir::RankedTensorType>();
    if (!tensorType) {
      throw std::runtime_error("Expected RankedTensorType");
    }
    auto elementType = tensorType.getElementType();
    return convertIntoDataType(elementType);
  });
}

size_t plaidml_expr_get_rank(  //
    plaidml_error* err,        //
    plaidml_expr* expr) {
  return ffi_wrap<size_t>(err, 0, [&] {
    auto tensorType = expr->value.getType().dyn_cast<mlir::RankedTensorType>();
    if (!tensorType) {
      throw std::runtime_error("Expected RankedTensorType");
    }
    return tensorType.getRank();
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
    size_t rank,              //
    plaidml_dim_expr** dims) {
  return ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_bind_dims> " << mlir::debugString(expr->value));
    llvm::SmallVector<mlir::Value*, 6> into;
    for (size_t i = 0; i < rank; i++) {
      IVLOG(3, "bind_dims> i: " << i << ", from: " << mlir::debugString(expr->value)
                                << ", into: " << mlir::debugString(dims[i]->value));
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
    return new plaidml_string{mlir::debugString(expr->value)};
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
  return ffi_wrap_void(err, [&] { GlobalContext::get()->BindBuffer(expr->value, buffer->buffer); });
}

plaidml_expr* plaidml_expr_clone(  //
    plaidml_error* err,            //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_clone> " << mlir::debugString(expr->value));
    return new plaidml_expr{expr->value};
  });
}

plaidml_dim_expr* plaidml_expr_get_dim(  //
    plaidml_error* err,                  //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_get_dim> " << mlir::debugString(expr->value));
    return new plaidml_dim_expr{expr->value};
  });
}

plaidml_expr* plaidml_expr_uint(  //
    plaidml_error* err,           //
    uint64_t value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_uint> " << value);
    return new plaidml_expr{GlobalContext::get()->MakeScalarConstantOp(value)};
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

plaidml_expr* plaidml_expr_float(  //
    plaidml_error* err,            //
    double value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_float");
    return new plaidml_expr{GlobalContext::get()->MakeScalarConstantOp(value)};
  });
}

plaidml_expr* plaidml_expr_cast(  //
    plaidml_error* err,           //
    plaidml_expr* tensor,         //
    plaidml_datatype dtype) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_cast");
    auto ctx = GlobalContext::get();
    auto targetType = convertFromDataType(dtype, ctx->getContext());
    return new plaidml_expr{ctx->MakeCastOp(tensor->value, targetType)};
  });
}

plaidml_expr* plaidml_expr_trace(  //
    plaidml_error* err,            //
    plaidml_expr* tensor,          //
    const char* msg) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_trace");
    return new plaidml_expr{GlobalContext::get()->MakeTraceOp(tensor->value, msg)};
  });
}

plaidml_expr* plaidml_expr_call(  //
    plaidml_error* err,           //
    const char* fn,               //
    size_t nargs,                 //
    plaidml_expr** args) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_call");
    std::vector<mlir::Value> values(nargs);
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
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(5, "plaidml_grad_override");
    // TODO(MLIR)
    return new plaidml_expr{out->value};
  });
}

plaidml_expr* plaidml_expr_index_map(  //
    plaidml_error* err,                //
    plaidml_expr* ref,                 //
    size_t rank,                       //
    plaidml_poly_expr** raw_idxs) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_index_map");
    std::vector<mlir::Value> idx_values(rank);
    for (size_t i = 0; i < rank; i++) {
      idx_values[i] = raw_idxs[i]->value;
    }
    if (ref) {
      return new plaidml_expr{GlobalContext::get()->MakeAffineTensorMapOp(ref->value, idx_values)};
    }
    return new plaidml_expr{GlobalContext::get()->MakeAffineMapOp(idx_values)};
  });
}

plaidml_expr* plaidml_expr_size_map(  //
    plaidml_error* err,               //
    size_t rank,                      //
    plaidml_dim_expr** raw_dims) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_size_map");
    std::vector<mlir::Value> dim_values;
    for (size_t i = 0; i < rank; i++) {
      dim_values.emplace_back(raw_dims[i]->value);
    }
    return new plaidml_expr{GlobalContext::get()->MakeAffineMapOp(dim_values)};
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
    std::vector<mlir::Value> src_values;
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

void plaidml_expr_contraction_set_no_reduce(  //
    plaidml_error* err,                       //
    plaidml_expr* expr,                       //
    bool no_reduce) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_contraction_set_no_reduce");
    if (!expr) {
      throw std::runtime_error("no_reduce can only be specified on a contraction.");
    }
    GlobalContext::get()->SetNoReduce(expr->value, no_reduce);
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
    std::vector<mlir::Value> wrt_values(nwrts);
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
  ffi_wrap_void(err, [&] {
    IVLOG(5, "plaidml_deriv_register> " << name);
    auto thunk = [](mlir::Value Y,                                //
                    mlir::Value dY,                               //
                    const llvm::SmallVector<mlir::Value, 3>& Xs,  //
                    void* user_fn,                                //
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
      llvm::SmallVector<mlir::Value, 3> ret(Xs.size());
      for (size_t i = 0; i < ret.size(); i++) {
        ret[i] = dX_exprs[i]->value;
        delete dX_exprs[i];  // TODO: Do I want this delete?
      }
      return ret;
    };
    DerivRegistry::Instance()->Register(name, thunk, reinterpret_cast<void*>(fn), user_ctx);
  });
}

void plaidml_poly_expr_free(plaidml_error* err, plaidml_poly_expr* expr) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_poly_expr_free> " << mlir::debugString(expr->value));
    GlobalContext::get()->Destroy(expr->value);
    delete expr;
  });
}

plaidml_string* plaidml_poly_expr_repr(  //
    plaidml_error* err,                  //
    plaidml_poly_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_repr");
    return new plaidml_string{mlir::debugString(expr->value)};
  });
}

plaidml_poly_expr* plaidml_poly_expr_dim(  //
    plaidml_error* err,                    //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_dim");
    return new plaidml_poly_expr{expr->value};
  });
}

plaidml_poly_expr* plaidml_poly_expr_index(  //
    plaidml_error* err,                      //
    const char* name) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_index");
    return new plaidml_poly_expr{GlobalContext::get()->MakePolyIndexOp(name)};
  });
}

plaidml_poly_expr* plaidml_poly_expr_literal(  //
    plaidml_error* err,                        //
    int64_t value) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_literal> " << value);
    return new plaidml_poly_expr{GlobalContext::get()->MakeConstantOp(value)};
  });
}

plaidml_poly_expr* plaidml_poly_expr_op(  //
    plaidml_error* err,                   //
    plaidml_int_op op,                    //
    size_t nargs,                         //
    plaidml_poly_expr** args) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_op> " << op);
    std::vector<mlir::Value> values(nargs);
    for (size_t i = 0; i < nargs; i++) {
      values[i] = args[i]->value;
    }
    return new plaidml_poly_expr{MakePolyOp(op, values)};
  });
}

void plaidml_dim_expr_free(  //
    plaidml_error* err,      //
    plaidml_dim_expr* expr) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_dim_expr_free> " << mlir::debugString(expr->value));
    GlobalContext::get()->Destroy(expr->value);
    delete expr;
  });
}

plaidml_string* plaidml_dim_expr_repr(  //
    plaidml_error* err,                 //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_repr");
    return new plaidml_string{mlir::debugString(expr->value)};
  });
}

plaidml_dim_expr* plaidml_dim_expr_none(  //
    plaidml_error* err                    //
) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_none");
    return new plaidml_dim_expr{GlobalContext::get()->MakeNoneOp()};
  });
}

plaidml_dim_expr* plaidml_dim_expr_int(  //
    plaidml_error* err,                  //
    int64_t value) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_int> " << value);
    return new plaidml_dim_expr{GlobalContext::get()->MakeConstantOp(value)};
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
    std::vector<mlir::Value> values(nargs);
    for (size_t i = 0; i < nargs; i++) {
      values[i] = args[i]->value;
    }
    return new plaidml_dim_expr{MakePolyOp(op, values)};
  });
}

void plaidml_tuple_free(  //
    plaidml_error* err,   //
    plaidml_tuple* tuple) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_tuple_free");
    for (size_t i = 0; i < tuple->size; i++) {
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
    size_t size,                     //
    plaidml_value** elts) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_tuple");
    Tuple tuple(size);
    for (size_t i = 0; i < size; i++) {
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
    auto size = tuple.size();
    auto elts = new plaidml_value*[size];
    for (size_t i = 0; i < size; i++) {
      elts[i] = new plaidml_value{tuple[i]->inner};
    }
    return new plaidml_tuple{size, elts};
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
            return mlir::debugString(arg.value);

          } else if constexpr (std::is_same_v<T, plaidml_expr>) {
            return mlir::debugString(arg.value);

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

plaidml_program* plaidml_compile(  //
    plaidml_error* err,            //
    const char* name,              //
    const char* target,            //
    size_t noutputs,               //
    plaidml_expr** raw_outputs,    //
    size_t nupdates,               //
    plaidml_expr** src_updates,    //
    plaidml_expr** dst_updates,    //
    plaidml_datatype floatx,       //
    plaidml_datatype intx,         //
    bool debug,                    //
    plaidml_program_args** raw_args) {
  return ffi_wrap<plaidml_program*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_compile");
    IVLOG(5, "  plaidml_compile>> noutputs: " << noutputs << ", nupdates: " << nupdates);
    ProgramMutations mutations;
    for (size_t i = 0; i < noutputs; i++) {
      if (!raw_outputs[i]) {
        throw std::runtime_error("Undefined output in plaidml_compile");
      }
      mutations.outputs.emplace_back(raw_outputs[i]->value);
    }
    for (size_t i = 0; i < nupdates; i++) {
      if (!src_updates[i]) {
        throw std::runtime_error("Undefined update src in plaidml_compile");
      }
      if (!dst_updates[i]) {
        throw std::runtime_error("Undefined update dst in plaidml_compile");
      }
      mutations.updates.emplace(ProgramUpdate{src_updates[i]->value, dst_updates[i]->value});
    }

    auto ctx = GlobalContext::get();
    auto floatType = convertFromDataType(floatx, ctx->getContext());
    if (!floatType.isa<mlir::FloatType>()) {
      throw std::runtime_error("Invalid floatx in plaidml_compile");
    }

    auto intType = convertFromDataType(intx, ctx->getContext());
    if (!intType.isa<mlir::IntegerType>()) {
      throw std::runtime_error("Invalid intx in plaidml_compile");
    }

    auto program = ctx->MakeProgram(name, mutations, floatType, intType);
    auto ret = new plaidml_program{program};
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
    auto dumpDir = pmlc::util::getEnvVar("PLAIDML_DUMP");
    program->compile(target, /*collectPasses=*/debug, /*dumpDir=*/dumpDir);
    return ret;
  });
}

plaidml_string* plaidml_program_repr(  //
    plaidml_error* err,                //
    plaidml_program* program) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_program_repr");
    return new plaidml_string{program->program->tileIR};
  });
}

plaidml_kvps* plaidml_program_get_passes(  //
    plaidml_error* err,                    //
    plaidml_program* program) {
  return ffi_wrap<plaidml_kvps*>(err, nullptr, [&] {
    const auto& passes = program->program->passes;
    auto ret = new plaidml_kvps{passes.size(), new plaidml_kvp[passes.size()]};
    size_t i = 0;
    for (auto it = passes.begin(), eit = passes.end(); it != eit; ++it, ++i) {
      ret->elts[i].key = new plaidml_string{it->name};
      ret->elts[i].value = new plaidml_string{it->ir};
    }
    return ret;
  });
}

void plaidml_program_args_free(  //
    plaidml_error* err,          //
    plaidml_program_args* args) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_program_args_free");
    for (unsigned i = 0; i < args->size; i++) {
      delete args->elts[i].shape;
      delete args->elts[i].tensor;
      delete args->elts[i].buffer;
    }
    delete[] args->elts;
    delete args;
  });
}

plaidml_strings* plaidml_targets_get(  //
    plaidml_error* err) {
  return ffi_wrap<plaidml_strings*>(err, nullptr, [&] {
    const auto& targets = pmlc::compiler::listTargets();
    auto strs = new plaidml_string*[targets.size()];
    for (unsigned i = 0; i < targets.size(); i++) {
      strs[i] = new plaidml_string{targets[i].str()};
    }
    return new plaidml_strings{targets.size(), strs};
  });
}

}  // extern "C"
