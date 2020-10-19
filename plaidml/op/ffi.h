// Copyright 2019 Intel Corporation.

#pragma once

#include "plaidml/core/ffi.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

plaidml_value* plaidml_op_make(  //
    plaidml_error* err,          //
    const char* op_name,         //
    plaidml_value* value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
