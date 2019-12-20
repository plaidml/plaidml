// Copyright 2019 Intel Corporation.

#pragma once

/// @cond FFI

#include "plaidml2/core/ffi.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

void plaidml_op_init(  //
    plaidml_error* err);

plaidml_value* plaidml_op_make(  //
    plaidml_error* err,          //
    const char* op_name,         //
    plaidml_value* value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

/// @endcond FFI
