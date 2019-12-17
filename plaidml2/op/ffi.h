// Copyright 2019 Intel Corporation.

#pragma once

#if defined _WIN32 || defined __CYGWIN__
#ifdef PLAIDML_OP_DLL
#define PLAIDML_OP_API __declspec(dllexport)
#else
#define PLAIDML_OP_API __declspec(dllimport)
#endif
#elif __GNUC__ >= 4
#define PLAIDML_OP_API __attribute__((visibility("default")))
#else
#define PLAIDML_OP_API
#endif

#include "plaidml2/core/ffi.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

PLAIDML_OP_API void plaidml_op_init(  //
    plaidml_error* err);

PLAIDML_OP_API plaidml_value* plaidml_op_make(  //
    plaidml_error* err,                         //
    const char* op_name,                        //
    plaidml_value* value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
