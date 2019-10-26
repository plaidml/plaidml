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

// Ensures that the PlaidML operation library globals have been initialized.
//
// Using this API is optional: if it is not called before the first use of the operation library, the
// operation library will be automatically initialized at that time.  This API is provided for callers that
// wish to control initialization order (typically ensuring that all components are loaded prior to performing
// other expensive computations), or that need to observe initialization failures separately from execution
// failures.
PLAIDML_OP_API void plaidml_op_init(  //
    plaidml_error* err);

PLAIDML_OP_API plaidml_expr* plaidml_op_make(  //
    plaidml_error* err,                        //
    const char* op_name,                       //
    plaidml_expr* expr);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
