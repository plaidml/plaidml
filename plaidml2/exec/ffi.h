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

PLAIDML_EXEC_API size_t plaidml_device_list_count(  //
    plaidml_error* err);

PLAIDML_EXEC_API void plaidml_device_list(  //
    plaidml_error* err,                     //
    size_t ndevices,                        //
    plaidml_string** device_ids);

//
// Target
//

PLAIDML_EXEC_API size_t plaidml_target_list_count(  //
    plaidml_error* err);

PLAIDML_EXEC_API void plaidml_target_list(  //
    plaidml_error* err,                     //
    size_t ntargets,                        //
    plaidml_string** targets);

//
// Buffer
//

PLAIDML_EXEC_API void plaidml_buffer_free(  //
    plaidml_error* err,                     //
    plaidml_buffer* buffer);

PLAIDML_EXEC_API plaidml_buffer* plaidml_buffer_alloc(  //
    plaidml_error* err,                                 //
    const char* device_id,                              //
    size_t size);

PLAIDML_EXEC_API plaidml_view* plaidml_buffer_mmap_current(  //
    plaidml_error* err,                                      //
    plaidml_buffer* buffer);

PLAIDML_EXEC_API plaidml_view* plaidml_buffer_mmap_discard(  //
    plaidml_error* err,                                      //
    plaidml_buffer* buffer);

//
// View
//

PLAIDML_EXEC_API void plaidml_view_free(  //
    plaidml_error* err,                   //
    plaidml_view* view);

PLAIDML_EXEC_API char* plaidml_view_data(  //
    plaidml_error* err,                    //
    plaidml_view* view);

PLAIDML_EXEC_API size_t plaidml_view_size(  //
    plaidml_error* err,                     //
    plaidml_view* view);

PLAIDML_EXEC_API void plaidml_view_writeback(  //
    plaidml_error* err,                        //
    plaidml_view* view);

//
// Executable
//

PLAIDML_EXEC_API plaidml_executable* plaidml_compile(  //
    plaidml_error* err,                                //
    plaidml_program* program,                          //
    const char* device_id,                             //
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
