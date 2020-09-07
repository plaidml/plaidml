// Copyright 2019 Intel Corporation.

#pragma once

#include "plaidml/core/ffi.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

void plaidml_exec_init(  //
    plaidml_error* err);

//
// Device
//

plaidml_strings* plaidml_devices_get(  //
    plaidml_error* err);

//
// Executable
//

plaidml_executable* plaidml_jit(  //
    plaidml_error* err,           //
    plaidml_program* program,     //
    const char* deviceID,         //
    size_t ninputs,               //
    plaidml_binding** inputs,     //
    size_t noutputs,              //
    plaidml_binding** outputs);

void plaidml_executable_free(  //
    plaidml_error* err,        //
    plaidml_executable* exec);

void plaidml_executable_run(  //
    plaidml_error* err,       //
    plaidml_executable* exec);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
