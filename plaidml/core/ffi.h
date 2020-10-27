// Copyright 2020 Intel Corporation

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

//
// Core API
//

typedef struct plaidml_string plaidml_string;
typedef struct plaidml_shape plaidml_shape;

typedef struct {
  size_t size;
  plaidml_string** elts;
} plaidml_strings;

typedef struct {
  size_t size;
  int64_t* elts;
} plaidml_integers;

typedef struct {
  plaidml_string* key;
  plaidml_string* value;
} plaidml_kvp;

typedef struct {
  size_t size;
  plaidml_kvp* elts;
} plaidml_kvps;

typedef struct {
  size_t size;
  plaidml_shape** elts;
} plaidml_shapes;

//
// Execution
//

typedef struct plaidml_buffer plaidml_buffer;
typedef struct plaidml_executable plaidml_executable;

//
// EDSL
//

typedef struct plaidml_builder plaidml_builder;
typedef struct plaidml_contraction plaidml_contraction;
typedef struct plaidml_dim_expr plaidml_dim_expr;
typedef struct plaidml_expr plaidml_expr;
typedef struct plaidml_poly_expr plaidml_poly_expr;
typedef struct plaidml_program plaidml_program;
typedef struct plaidml_value plaidml_value;

//
// String
//

const char* plaidml_string_ptr(  //
    plaidml_string* str);

void plaidml_string_free(  //
    plaidml_string* str);

//
// Error
//

typedef struct {
  size_t code;
  plaidml_string* msg;
} plaidml_error;

//
// Library
//

void plaidml_init(  //
    plaidml_error* err);

void plaidml_shutdown(  //
    plaidml_error* err);

const char* plaidml_version(  //
    plaidml_error* err);

void plaidml_integers_free(  //
    plaidml_error* err,      //
    plaidml_integers* ints);

void plaidml_strings_free(  //
    plaidml_error* err,     //
    plaidml_strings* strs);

void plaidml_kvps_free(  //
    plaidml_error* err,  //
    plaidml_kvps* kvps);

void plaidml_shapes_free(  //
    plaidml_error* err,    //
    plaidml_shapes* shapes);

plaidml_kvps* plaidml_settings_list(  //
    plaidml_error* err                //
);

plaidml_string* plaidml_settings_get(  //
    plaidml_error* err,                //
    const char* key);

void plaidml_settings_set(  //
    plaidml_error* err,     //
    const char* key,        //
    const char* value);

void plaidml_settings_load(  //
    plaidml_error* err);

void plaidml_settings_save(  //
    plaidml_error* err);

//
// Shape
//

typedef enum {
  PLAIDML_DATA_INVALID = 0,
  PLAIDML_DATA_INTX = 1,
  PLAIDML_DATA_UINTX = 2,
  PLAIDML_DATA_FLOATX = 3,
  PLAIDML_DATA_BOOLEAN = 4,
  PLAIDML_DATA_INT8 = 5,
  PLAIDML_DATA_UINT8 = 6,
  PLAIDML_DATA_INT16 = 7,
  PLAIDML_DATA_UINT16 = 8,
  PLAIDML_DATA_INT32 = 9,
  PLAIDML_DATA_UINT32 = 10,
  PLAIDML_DATA_INT64 = 11,
  PLAIDML_DATA_UINT64 = 12,
  PLAIDML_DATA_BFLOAT16 = 13,
  PLAIDML_DATA_FLOAT16 = 14,
  PLAIDML_DATA_FLOAT32 = 15,
  PLAIDML_DATA_FLOAT64 = 16,
} plaidml_datatype;

void plaidml_shape_free(  //
    plaidml_error* err,   //
    plaidml_shape* shape);

plaidml_shape* plaidml_shape_alloc(  //
    plaidml_error* err,              //
    plaidml_datatype dtype,          //
    size_t rank,                     //
    const int64_t* sizes,            //
    const int64_t* strides);

plaidml_shape* plaidml_shape_clone(  //
    plaidml_error* err,              //
    plaidml_shape* shape);

plaidml_string* plaidml_shape_repr(  //
    plaidml_error* err,              //
    plaidml_shape* shape);

size_t plaidml_shape_get_rank(  //
    plaidml_error* err,         //
    plaidml_shape* shape);

plaidml_datatype plaidml_shape_get_dtype(  //
    plaidml_error* err,                    //
    plaidml_shape* shape);

plaidml_integers* plaidml_shape_get_sizes(  //
    plaidml_error* err,                     //
    plaidml_shape* shape);

plaidml_integers* plaidml_shape_get_strides(  //
    plaidml_error* err,                       //
    plaidml_shape* shape);

uint64_t plaidml_shape_get_nbytes(  //
    plaidml_error* err,             //
    plaidml_shape* shape);

//
// Buffer
//

plaidml_buffer* plaidml_buffer_alloc(  //
    plaidml_error* err,                //
    plaidml_shape* shape               //
);

plaidml_buffer* plaidml_buffer_adopt(  //
    plaidml_error* err,                //
    plaidml_shape* shape,              //
    char* data,                        //
    size_t size);

void plaidml_buffer_free(  //
    plaidml_error* err,    //
    plaidml_buffer* buffer);

plaidml_buffer* plaidml_buffer_clone(  //
    plaidml_error* err,                //
    plaidml_buffer* buffer);

plaidml_shape* plaidml_buffer_shape(  //
    plaidml_error* err,               //
    plaidml_buffer* buffer);

char* plaidml_buffer_data(  //
    plaidml_error* err,     //
    plaidml_buffer* buffer);

size_t plaidml_buffer_size(  //
    plaidml_error* err,      //
    plaidml_buffer* buffer);

//
// plaidml_program
//

void plaidml_program_free(  //
    plaidml_error* err,     //
    plaidml_program* program);

plaidml_string* plaidml_program_repr(  //
    plaidml_error* err,                //
    plaidml_program* program);

plaidml_shapes* plaidml_program_get_inputs(  //
    plaidml_error* err,                      //
    plaidml_program* program);

plaidml_shapes* plaidml_program_get_outputs(  //
    plaidml_error* err,                       //
    plaidml_program* program);

plaidml_kvps* plaidml_program_get_passes(  //
    plaidml_error* err,                    //
    plaidml_program* program);

void plaidml_program_compile(  //
    plaidml_error* err,        //
    plaidml_program* program,  //
    bool debug,                //
    const char* target);

plaidml_buffer* plaidml_program_save(  //
    plaidml_error* err,                //
    plaidml_program* program);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
