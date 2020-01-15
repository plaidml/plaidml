// Copyright 2019 Intel Corporation.

#pragma once

// This is the PlaidML OpenCL shim.
//
// OpenCL has several versions; as of this writing, 1.2 and 2.2 are
// fairly common.  The various versions add useful functionality, and
// add additional APIs for accessing that functionality.
//
// To use OpenCL, we need both headers and libraries.
//
// To get headers: on some systems (e.g. macOS), we use the
// system-provided headers to ensure compatibility with what's shipped
// with the system.  On other systems, we use headers provided by the
// Khronos Group (R).
//
// The system-provided headers may not contain definitions for later
// versions of OpenCL.  The Khronos headers are typically newer, and
// contain all of the definitions, even if the system libraries do not
// provide symbols for those definitions.
//
// To get libraries: on some systems (e.g. macOS and Windows), we use
// the system-provided libraries to ensure compatibility with what's
// shipped with the system (we've seen subtle issues when we've built
// our own).  On other systems, we use the libraries that happen to be
// installed, which could be anything from OpenCL 1.0 on up depending
// on the installation.
//
// Just to complicate things further, OpenCL supports the notion of an
// installable client driver (ICD) loader, a shim library that
// provides the OpenCL API symbols and forwards calls to the
// appropriate driver-specific OpenCL implementation.  Devices may
// support any version of OpenCL, independent of the ICD loader.
//
// And the last little wrinkle is that on some platforms (e.g. Linux),
// we don't want to require that the OpenCL libraries be installed at
// all; we want everything to be dynamically loaded and bound.  On
// other platforms (e.g. macOS and Windows), we never want to
// dynamically load the libraries; we want to link against the shared
// libraries directly.
//
// To implement all this:
//
// This header and library target does whatever platform-specific
// magic is required in order to use OpenCL.
//
// To simplify callers, the caller interface is a direct translation
// of the OpenCL call: `clSomeAPICall()` becomes
// `vertexai::shim::opencl::SomeAPICall()`.  We recommend that callers
// use `namespace ocl = vertexai::shim::opencl`.
//
// Each API function is implemented one of three ways, depending on
// the platform and the version of OpenCL that supports the API
// function:
//
//   ) Static/Shared load: call the OpenCL symbol directly
//   ) Dynamic load: load the OpenCL symbol and call it as needed
//   ) Always unimplemented
//
// To implement each API:
//
//   ) We explicitly code each API implementation, instead of trying
//     to do something clever with macro expansion; we think it's a
//     bit more readable/debuggable, even though it's also a bit
//     monotonous.
//
//   ) We separate interface and implementation, partially for
//     clarity, and partially since we're leveraging
//     statically-initialized pointers and char strings, and we want
//     to ensure that there's only a single definition of each across
//     all translation units in the program (which most compilers
//     should handle just fine, but it's easy to be sure).  This also
//     keeps macro pollution out of users of the header.
//
//   ) We use a use a per-OpenCL-version macro flag to control whether
//     OpenCL API calls for that version should be static or not.
//
//   ) Static calls are just made immediately.
//
//   ) Non-static calls declare a statically-initialized
//     implementation function pointer.
//
//     If dynamic loading is available, the static initializer loads
//     the library and returns the requested pointer.
//
//     Otherwise, the static initializer throws an ApiUnavailable
//     exception.  This causes the implementation function pointer
//     to not be initialized.
//
// As an aside: we did consider defining weak symbols for OpenCL calls
// and letting the system dynamic linker bind what it could.  We
// rejected that plan because we'd rather not depend on the system
// implementation of weak symbols; it's good to be explicit about
// what's going on.

#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

#ifndef CL_VERSION_2_0
#define CL_MEM_SVM_FINE_GRAIN_BUFFER 0
typedef cl_bitfield cl_svm_mem_flags;
#endif  // !defined(CL_VERSION_2_0)

#include <string>
#include <utility>

#include "base/util/error.h"

namespace vertexai {
namespace shim {
namespace opencl {

// The error thrown by OpenCL APIs not available on the current system.
//
// This occurs either because we don't implement it on the current
// platform, or because the system we're running on doesn't have a
// version of OpenCL that provides the API.
class ApiUnavailable : public error::Unimplemented {
 public:
  explicit ApiUnavailable(std::string msg) noexcept : Unimplemented{std::move(msg)} {}
};

extern cl_int BuildProgram(cl_program program, cl_uint num_devices, const cl_device_id* device_list,
                           const char* options, void(CL_CALLBACK* pfn_notify)(cl_program, void*), void* user_data);

extern cl_mem CreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret);

extern cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device,
                                           cl_command_queue_properties properties, cl_int* errcode_ret);

extern cl_context CreateContext(const cl_context_properties* properties, cl_uint num_devices,
                                const cl_device_id* devices,
                                void(CL_CALLBACK* pfn_notify)(const char*, const void*, size_t, void*), void* user_data,
                                cl_int* errcode_ret);

extern cl_kernel CreateKernel(cl_program program, const char* kernel_name, cl_int* errcode_ret);

extern cl_program CreateProgramWithSource(cl_context context, cl_uint count, const char** strings,
                                          const size_t* lengths, cl_int* errcode_ret);

extern cl_program CreateProgramWithIL(cl_context context, const void* il, const size_t length, cl_int* errcode_ret);

extern cl_mem CreateSubBuffer(cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type buffer_create_type,
                              const void* buffer_create_info, cl_int* errcode_ret);

extern cl_event CreateUserEvent(cl_context context, cl_int* errcode_ret);

extern cl_int EnqueueCopyBuffer(cl_command_queue queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset,
                                size_t dst_offset, size_t size, cl_uint num_events_in_wait_list,
                                const cl_event* event_wait_list, cl_event* event);

extern cl_int EnqueueFillBuffer(cl_command_queue queue, cl_mem buffer, const void* pattern, size_t pattern_size,
                                size_t offset, size_t size, cl_uint num_events_in_wait_list,
                                const cl_event* event_wait_list, cl_event* event);

extern void* EnqueueMapBuffer(cl_command_queue queue, cl_mem buffer, cl_bool blocking_map, cl_map_flags map_flags,
                              size_t offset, size_t size, cl_uint num_events_in_wait_list,
                              const cl_event* event_wait_list, cl_event* event, cl_int* errcode_ret);

extern cl_int EnqueueMarkerWithWaitList(cl_command_queue queue, cl_uint num_events_in_wait_list,
                                        const cl_event* event_wait_list, cl_event* event);

extern cl_int EnqueueNDRangeKernel(cl_command_queue queue, cl_kernel kernel, cl_uint work_dim,
                                   const size_t* global_work_offset, const size_t* global_work_size,
                                   const size_t* local_work_size, cl_uint num_events_in_wait_list,
                                   const cl_event* event_wait_list, cl_event* event);

extern cl_int EnqueueReadBuffer(cl_command_queue queue, cl_mem buffer, bool blocking_read, size_t offset, size_t size,
                                void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                                cl_event* event);

extern cl_int EnqueueSVMMemFill(cl_command_queue queue, void* svm_ptr, const void* pattern, size_t pattern_size,
                                size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                                cl_event* event);

extern cl_int EnqueueUnmapMemObject(cl_command_queue queue, cl_mem memobj, void* mapped_ptr,
                                    cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);

extern cl_int EnqueueWriteBuffer(cl_command_queue queue, cl_mem buffer, bool blocking_write, size_t offset, size_t size,
                                 const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                                 cl_event* event);

extern cl_int Finish(cl_command_queue queue);
extern cl_int Flush(cl_command_queue queue);

extern cl_int GetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries,
                           cl_device_id* devices, cl_uint* num_devices);

extern cl_int GetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void* param_value,
                            size_t* param_value_size_ret);

extern cl_int GetEventInfo(cl_event event, cl_event_info param_name, size_t param_value_size, void* param_value,
                           size_t* param_value_size_ret);

extern cl_int GetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size,
                                    void* param_value, size_t* param_value_size_ret);

extern cl_int GetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name,
                                     size_t param_value_size, void* param_value, size_t* param_value_size_ret);

extern cl_int GetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms);

extern cl_int GetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size,
                              void* param_value, size_t* param_value_size_ret);

extern cl_int GetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name,
                                  size_t param_value_size, void* param_value, size_t* param_value_size_ret);

extern cl_int GetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void* param_value,
                             size_t* param_value_size_ret);

extern cl_int ReleaseContext(cl_context c);
extern cl_int ReleaseCommandQueue(cl_command_queue c);
extern cl_int ReleaseEvent(cl_event e);
extern cl_int ReleaseKernel(cl_kernel k);
extern cl_int ReleaseMemObject(cl_mem m);
extern cl_int ReleaseProgram(cl_program p);

extern cl_int RetainContext(cl_context c);
extern cl_int RetainCommandQueue(cl_command_queue c);
extern cl_int RetainEvent(cl_event e);
extern cl_int RetainKernel(cl_kernel k);
extern cl_int RetainMemObject(cl_mem m);
extern cl_int RetainProgram(cl_program p);

extern cl_int SetEventCallback(cl_event event, cl_int command_exec_callback_type,
                               void(CL_CALLBACK* pfn_notify)(cl_event, cl_int, void*), void* user_data);

extern cl_int SetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value);
extern cl_int SetKernelArgSVMPointer(cl_kernel kernel, cl_uint arg_index, const void* arg_value);
extern cl_int SetUserEventStatus(cl_event event, cl_int execution_status);

extern void* SVMAlloc(cl_context context, cl_svm_mem_flags flags, size_t size, cl_uint alignment);

extern void SVMFree(cl_context context, void* svm_pointer);

}  // namespace opencl
}  // namespace shim
}  // namespace vertexai
