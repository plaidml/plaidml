// Copyright 2019 Intel Corporation.

#pragma once

#include "plaidml2/core/ffi.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  PLAIDML_INT_OP_NEG,
  PLAIDML_INT_OP_ADD,
  PLAIDML_INT_OP_SUB,
  PLAIDML_INT_OP_MUL,
  PLAIDML_INT_OP_DIV,
  PLAIDML_INT_OP_MAX,
  PLAIDML_INT_OP_MIN,
} plaidml_int_op;

typedef enum {
  PLAIDML_VALUE_NONE,
  PLAIDML_VALUE_DIM,
  PLAIDML_VALUE_EXPR,
  PLAIDML_VALUE_FLOAT,
  PLAIDML_VALUE_INT,
  PLAIDML_VALUE_STR,
  PLAIDML_VALUE_TUPLE,
} plaidml_value_kind;

void plaidml_edsl_init(  //
    plaidml_error* err);

//
// plaidml_logical_shape
//

plaidml_logical_shape* plaidml_logical_shape_alloc(  //
    plaidml_error* err,                              //
    plaidml_datatype dtype,                          //
    size_t ndims,                                    //
    const int64_t* dims);

void plaidml_logical_shape_free(  //
    plaidml_error* err,           //
    plaidml_logical_shape* shape);

plaidml_logical_shape* plaidml_logical_shape_clone(  //
    plaidml_error* err,                              //
    plaidml_logical_shape* shape);

plaidml_shape* plaidml_logical_shape_into_tensor_shape(  //
    plaidml_error* err,                                  //
    plaidml_logical_shape* shape);                       //

plaidml_string* plaidml_logical_shape_repr(  //
    plaidml_error* err,                      //
    plaidml_logical_shape* shape);

size_t plaidml_logical_shape_get_ndims(  //
    plaidml_error* err,                  //
    plaidml_logical_shape* shape);

plaidml_datatype plaidml_logical_shape_get_dtype(  //
    plaidml_error* err,                            //
    plaidml_logical_shape* shape);

int64_t plaidml_logical_shape_get_dim_int(  //
    plaidml_error* err,                     //
    plaidml_logical_shape* shape,           //
    size_t dim);

//
// plaidml_poly_expr
//

void plaidml_poly_expr_free(  //
    plaidml_error* err,       //
    plaidml_poly_expr* expr);

plaidml_string* plaidml_poly_expr_repr(  //
    plaidml_error* err,                  //
    plaidml_poly_expr* expr);

plaidml_poly_expr* plaidml_poly_expr_dim(  //
    plaidml_error* err,                    //
    plaidml_dim_expr* expr);

plaidml_poly_expr* plaidml_poly_expr_index(  //
    plaidml_error* err,                      //
    const char* name);

plaidml_poly_expr* plaidml_poly_expr_literal(  //
    plaidml_error* err,                        //
    int64_t value);

plaidml_poly_expr* plaidml_poly_expr_op(  //
    plaidml_error* err,                   //
    plaidml_int_op op,                    //
    size_t nargs,                         //
    plaidml_poly_expr** args);

//
// plaidml_dim_expr
//

void plaidml_dim_expr_free(  //
    plaidml_error* err,      //
    plaidml_dim_expr* expr);

plaidml_string* plaidml_dim_expr_repr(  //
    plaidml_error* err,                 //
    plaidml_dim_expr* expr);

plaidml_dim_expr* plaidml_dim_expr_none(  //
    plaidml_error* err                    //
);

plaidml_dim_expr* plaidml_dim_expr_int(  //
    plaidml_error* err,                  //
    int64_t value);

int64_t plaidml_dim_expr_get_int(  //
    plaidml_error* err,            //
    plaidml_dim_expr* expr);

plaidml_dim_expr* plaidml_dim_expr_op(  //
    plaidml_error* err,                 //
    plaidml_int_op op,                  //
    size_t nargs,                       //
    plaidml_dim_expr** args);

//
// plaidml_deriv
//

void plaidml_expr_gradient(  //
    plaidml_error* err,      //
    size_t nwrts,            //
    plaidml_expr** wrts,     //
    plaidml_expr* loss,      //
    plaidml_expr** derivs);

typedef void (*plaidml_deriv)(  //
    void* user_ctx,             //
    plaidml_expr* Y,            //
    plaidml_expr* dY,           //
    size_t nXs,                 //
    plaidml_expr** Xs,          //
    plaidml_expr** dXs);

void plaidml_deriv_register(  //
    plaidml_error* err,       //
    const char* name,         //
    plaidml_deriv fn,         //
    void* user_ctx);

//
// plaidml_expr
//

void plaidml_expr_free(  //
    plaidml_error* err,  //
    plaidml_expr* expr);

void* plaidml_expr_ptr(  //
    plaidml_error* err,  //
    plaidml_expr* expr);

plaidml_logical_shape* plaidml_expr_get_shape(  //
    plaidml_error* err,                         //
    plaidml_expr* expr);

void plaidml_expr_bind_shape(  //
    plaidml_error* err,        //
    plaidml_expr* expr,        //
    plaidml_logical_shape* shape);

void plaidml_expr_bind_dims(  //
    plaidml_error* err,       //
    plaidml_expr* expr,       //
    size_t ndims,             //
    plaidml_dim_expr** dims);

plaidml_string* plaidml_expr_repr(  //
    plaidml_error* err,             //
    plaidml_expr* expr);

plaidml_expr* plaidml_expr_clone(  //
    plaidml_error* err,            //
    plaidml_expr* expr);

plaidml_dim_expr* plaidml_expr_get_dim(  //
    plaidml_error* err,                  //
    plaidml_expr* expr);

plaidml_expr* plaidml_expr_dim(  //
    plaidml_error* err,          //
    plaidml_dim_expr* expr);

plaidml_expr* plaidml_expr_placeholder(  //
    plaidml_error* err,                  //
    plaidml_logical_shape* shape,        //
    plaidml_buffer* buffer,              //
    const char* name);

void plaidml_expr_param_reset(  //
    plaidml_error* err,         //
    plaidml_expr* shape,        //
    plaidml_buffer* buffer);

plaidml_expr* plaidml_expr_int(  //
    plaidml_error* err,          //
    int64_t value);

plaidml_expr* plaidml_expr_float(  //
    plaidml_error* err,            //
    double value);

plaidml_expr* plaidml_expr_call(  //
    plaidml_error* err,           //
    const char* fn,               //
    size_t nargs,                 //
    plaidml_expr** args);

plaidml_expr* plaidml_expr_cast(  //
    plaidml_error* err,           //
    plaidml_expr* tensor,         //
    plaidml_datatype dtype);

plaidml_expr* plaidml_expr_index_map(  //
    plaidml_error* err,                //
    plaidml_expr* ref,                 //
    size_t ndims,                      //
    plaidml_poly_expr** idxs);

plaidml_expr* plaidml_expr_size_map(  //
    plaidml_error* err,               //
    size_t ndims,                     //
    plaidml_dim_expr** sizes);

plaidml_expr* plaidml_expr_grad_override(  //
    plaidml_error* err,                    //
    plaidml_deriv fn,                      //
    void* user_ctx,                        //
    size_t nins,                           //
    plaidml_expr** ins,                    //
    plaidml_expr* out);

typedef enum {
  PLAIDML_AGG_OP_NONE,
  PLAIDML_AGG_OP_SUM,
  PLAIDML_AGG_OP_MAX,
  PLAIDML_AGG_OP_MIN,
  PLAIDML_AGG_OP_PROD,
  PLAIDML_AGG_OP_ASSIGN
} plaidml_agg_op;

typedef enum {
  PLAIDML_COMBO_OP_NONE,
  PLAIDML_COMBO_OP_MUL,
  PLAIDML_COMBO_OP_ADD,
  PLAIDML_COMBO_OP_EQ,
  PLAIDML_COMBO_OP_COND,
} plaidml_combo_op;

plaidml_expr* plaidml_expr_contraction(  //
    plaidml_error* err,                  //
    plaidml_agg_op agg_op,               //
    plaidml_combo_op combo_op,           //
    plaidml_expr* sink_idxs,             //
    plaidml_expr* sink_sizes,            //
    size_t nsrcs,                        //
    plaidml_expr** src_idxs,             //
    const char* name);

void plaidml_expr_contraction_add_constraint(  //
    plaidml_error* err,                        //
    plaidml_expr* expr,                        //
    plaidml_poly_expr* lhs,                    //
    plaidml_dim_expr* rhs);

void plaidml_expr_contraction_set_no_reduce(  //
    plaidml_error* err,                       //
    plaidml_expr* expr,                       //
    bool no_reduce);

void plaidml_expr_contraction_set_use_default(  //
    plaidml_error* err,                         //
    plaidml_expr* expr,                         //
    plaidml_expr* use_default);

//
// plaidml_value
//

typedef struct plaidml_tuple {
  size_t nelts;
  plaidml_value** elts;
} plaidml_tuple;

void plaidml_tuple_free(  //
    plaidml_error* err,   //
    plaidml_tuple* tuple);

void plaidml_value_free(  //
    plaidml_error* err,   //
    plaidml_value* value);

plaidml_value* plaidml_value_clone(  //
    plaidml_error* err,              //
    plaidml_value* value);

plaidml_value_kind plaidml_value_get_kind(  //
    plaidml_error* err,                     //
    plaidml_value* value);

plaidml_value* plaidml_value_none(  //
    plaidml_error* err              //
);

plaidml_value* plaidml_value_dim(  //
    plaidml_error* err,            //
    plaidml_dim_expr* expr);

plaidml_value* plaidml_value_expr(  //
    plaidml_error* err,             //
    plaidml_expr* expr);

plaidml_value* plaidml_value_float(  //
    plaidml_error* err,              //
    double value);

plaidml_value* plaidml_value_int(  //
    plaidml_error* err,            //
    int64_t value);

plaidml_value* plaidml_value_str(  //
    plaidml_error* err,            //
    const char* value);

plaidml_value* plaidml_value_tuple(  //
    plaidml_error* err,              //
    size_t nelts,                    //
    plaidml_value** elts);

plaidml_dim_expr* plaidml_value_dim_get(  //
    plaidml_error* err,                   //
    plaidml_value* value);

plaidml_expr* plaidml_value_expr_get(  //
    plaidml_error* err,                //
    plaidml_value* value);

double plaidml_value_float_get(  //
    plaidml_error* err,          //
    plaidml_value* value);

int64_t plaidml_value_int_get(  //
    plaidml_error* err,         //
    plaidml_value* value);

plaidml_string* plaidml_value_str_get(  //
    plaidml_error* err,                 //
    plaidml_value* value);

plaidml_tuple* plaidml_value_tuple_get(  //
    plaidml_error* err,                  //
    plaidml_value* value);

plaidml_string* plaidml_value_repr(  //
    plaidml_error* err,              //
    plaidml_value* value);

//
// plaidml_program
//

typedef struct plaidml_program_arg {
  bool is_input;
  plaidml_expr* tensor;
  plaidml_logical_shape* shape;
  plaidml_buffer* buffer;
} plaidml_program_arg;

typedef struct plaidml_program_args {
  size_t nargs;
  plaidml_program_arg* args;
} plaidml_program_args;

void plaidml_program_free(  //
    plaidml_error* err,     //
    plaidml_program* program);

plaidml_program* plaidml_program_evaluate(  //
    plaidml_error* err,                     //
    const char* name,                       //
    size_t noutputs,                        //
    plaidml_expr** outputs,                 //
    size_t nupdates,                        //
    plaidml_expr** src_updates,             //
    plaidml_expr** dst_updates,             //
    plaidml_program_args** args);

plaidml_string* plaidml_program_repr(  //
    plaidml_error* err,                //
    plaidml_program* program);

void plaidml_program_args_free(  //
    plaidml_error* err,          //
    plaidml_program_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
