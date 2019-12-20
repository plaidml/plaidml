// Copyright 2019 Intel Corporation.

#pragma once

#include "plaidml2/core/ffi.h"

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
// Target
//

plaidml_strings* plaidml_targets_get(  //
    plaidml_error* err);

//
// Executable
//

plaidml_executable* plaidml_compile(  //
    plaidml_error* err,               //
    plaidml_program* program,         //
    const char* device,               //
    const char* target,               //
    size_t ninputs,                   //
    plaidml_binding** inputs,         //
    size_t noutputs,                  //
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
