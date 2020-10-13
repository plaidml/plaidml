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
    const char* device);

void plaidml_executable_free(  //
    plaidml_error* err,        //
    plaidml_executable* exec);

void plaidml_executable_run(   //
    plaidml_error* err,        //
    plaidml_executable* exec,  //
    size_t ninputs,            //
    plaidml_buffer** inputs,   //
    size_t noutputs,           //
    plaidml_buffer** outputs);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
