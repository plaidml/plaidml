// Copyright 2019 Intel Corporation.

#pragma once

#if defined _WIN32 || defined __CYGWIN__
#ifdef PLAIDML_EXEC_DLL
#define PLAIDML_EXEC_API __declspec(dllexport)
#else
#define PLAIDML_EXEC_API __declspec(dllimport)
#endif
#elif __GNUC__ >= 4
#define PLAIDML_EXEC_API __attribute__((visibility("default")))
#else
#define PLAIDML_EXEC_API
#endif

#include "plaidml2/core/ffi.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

PLAIDML_EXEC_API void plaidml_exec_init(  //
    plaidml_error* err);

//
// Device
//

PLAIDML_EXEC_API plaidml_strings* plaidml_devices_get(  //
    plaidml_error* err);

//
// Target
//

PLAIDML_EXEC_API plaidml_strings* plaidml_targets_get(  //
    plaidml_error* err);

//
// Executable
//

PLAIDML_EXEC_API plaidml_executable* plaidml_compile(  //
    plaidml_error* err,                                //
    plaidml_program* program,                          //
    const char* device,                                //
    const char* target,                                //
    size_t ninputs,                                    //
    plaidml_binding** inputs,                          //
    size_t noutputs,                                   //
    plaidml_binding** outputs);

PLAIDML_EXEC_API void plaidml_executable_free(  //
    plaidml_error* err,                         //
    plaidml_executable* exec);

PLAIDML_EXEC_API void plaidml_executable_run(  //
    plaidml_error* err,                        //
    plaidml_executable* exec);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
