// Copyright 2019 Intel Corporation.

#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#endif  // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  PLAIDML_DATA_INVALID = 0,
  PLAIDML_DATA_BOOLEAN = 0x02,
  PLAIDML_DATA_INT8 = 0x10,
  PLAIDML_DATA_INT16 = 0x11,
  PLAIDML_DATA_INT32 = 0x12,
  PLAIDML_DATA_INT64 = 0x13,
  PLAIDML_DATA_INT128 = 0x14,
  PLAIDML_DATA_UINT8 = 0x20,
  PLAIDML_DATA_UINT16 = 0x21,
  PLAIDML_DATA_UINT32 = 0x22,
  PLAIDML_DATA_UINT64 = 0x23,
  PLAIDML_DATA_FLOAT16 = 0x31,
  PLAIDML_DATA_FLOAT32 = 0x32,
  PLAIDML_DATA_FLOAT64 = 0x33,
  PLAIDML_DATA_PRNG = 0x40,
} plaidml_datatype;

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

typedef struct tile_string tile_string;

typedef struct {
  size_t code;
  tile_string* msg;
} tile_error;

typedef struct tile_shape tile_shape;

typedef struct tile_expr tile_expr;

typedef struct tile_poly_expr tile_poly_expr;

typedef struct tile_program tile_program;

const char* tile_string_ptr(tile_string* str);

void tile_string_free(tile_string* str);

tile_shape* tile_shape_alloc(tile_error* err, plaidml_datatype dtype, const char* layout);

tile_string* tile_shape_repr(tile_error* err, tile_shape* shape);

void tile_shape_add_dimension(tile_error* err, tile_shape* shape, uint64_t size, int64_t stride);

size_t tile_shape_get_rank(tile_error* err, tile_shape* shape);

plaidml_datatype tile_shape_get_type(tile_error* err, tile_shape* shape);

uint64_t tile_shape_get_dimension_size(tile_error* err, tile_shape* shape, size_t dim);

int64_t tile_shape_get_dimension_stride(tile_error* err, tile_shape* shape, size_t dim);

uint64_t tile_shape_get_byte_size(tile_error* err, tile_shape* shape);

const void* tile_shape_get_ptr(tile_error* err, tile_shape* shape);

void tile_shape_free(tile_error* err, tile_shape* shape);

void tile_expr_free(tile_error* err, tile_expr* expr);

tile_string* tile_expr_repr(tile_error* err, tile_expr* expr);

tile_expr* tile_expr_param(tile_error* err, tile_shape* shape, const char* name);

tile_expr* tile_expr_int(tile_error* err, int64_t value);

tile_expr* tile_expr_float(tile_error* err, double value);

tile_expr* tile_expr_call(tile_error* err, const char* fn, size_t nargs, tile_expr** args);

tile_expr* tile_expr_tensor_spec(tile_error* err,              //
                                 tile_expr* ref,               //
                                 size_t rank,                  //
                                 tile_poly_expr** input_idxs,  //
                                 size_t* output_sizes);

tile_expr* tile_expr_contraction(tile_error* err,         //
                                 tile_agg_op agg_op,      //
                                 tile_combo_op combo_op,  //
                                 tile_expr* raw_output,   //
                                 size_t ninputs,          //
                                 tile_expr** raw_inputs,  //
                                 const char* name);

void tile_expr_contraction_set_no_defract(tile_error* err, tile_expr* expr, bool no_defract);

void tile_expr_contraction_set_use_default(tile_error* err, tile_expr* expr, tile_expr* use_default);

tile_shape* tile_expr_evaluate_shape(tile_error* err, tile_expr* expr);

void tile_poly_expr_free(tile_error* err, tile_poly_expr* expr);

tile_string* tile_poly_expr_repr(tile_error* err, tile_poly_expr* expr);

tile_poly_expr* tile_poly_expr_index(tile_error* err, const char* name);

tile_poly_expr* tile_poly_expr_literal(tile_error* err, int64_t value);

tile_poly_expr* tile_poly_expr_op(tile_error* err, tile_poly_op op, size_t nargs, tile_poly_expr** args);

void tile_poly_expr_add_constraint(tile_error* err, tile_poly_expr* lhs, size_t rhs);

void tile_program_free(tile_error* err, tile_program* program);

tile_program* tile_program_evaluate(tile_error* err, const char* name, size_t nexprs, tile_expr** raw_exprs);

tile_string* tile_program_repr(tile_error* err, tile_program* program);

// This is a temporary HACK to provide underlying access to the RunInfo
const void* tile_program_runinfo(tile_error* err, tile_program* program);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
