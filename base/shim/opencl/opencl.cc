// Copyright 2019 Intel Corporation.

#include "base/shim/opencl/opencl.h"

#include "base/util/logging.h"

namespace vertexai {
namespace shim {
namespace opencl {

#ifdef __APPLE__

#define OCL_STATIC_1_0
#define OCL_STATIC_1_1
#define OCL_STATIC_1_2

namespace {

template <typename F>
F GetImpl(const char* symbol) {
  // For Apple platforms, any non-static API call (i.e. anything from
  // OpenCL 2.0 or later) is always unavailable.
  throw ApiUnavailable{symbol};
}

}  // namespace

#elif defined _WIN32 || defined __CYGWIN__

// On Windows, we use dynamic loading for everything.
// This way, we don't have to worry about having an OpenCL SDK installed,
// just the OpenCL runtime.

namespace {

HMODULE LoadOpenCL(const char* symbol) {
  // TODO: It'd be great to unload this library at some point.
  HMODULE lib = LoadLibrary("OpenCL");
  if (!lib) {
    throw ApiUnavailable{std::string(symbol)};
  }
  return lib;
}

HMODULE GetOpenCL(const char* symbol) {
  static HMODULE lib = LoadOpenCL(symbol);
  return lib;
}

template <typename F>
F GetImpl(const char* symbol) {
  void* impl = GetProcAddress(GetOpenCL(symbol), symbol);
  if (!impl) {
    throw ApiUnavailable{std::string(symbol)};
  }
  return reinterpret_cast<F>(impl);
}

}  // namespace

#else

// If this isn't an Apple or Windows platform, we use dynamic loading
// for everything, and we assume that we have a POSIX-compatible
// dlopen() implementation and Linux-compatible dynamic library
// naming.

}  // namespace opencl
}  // namespace shim
}  // namespace vertexai

#include <dlfcn.h>

namespace vertexai {
namespace shim {
namespace opencl {
namespace {

void* LoadOpenCL(const char* symbol) {
  // TODO: It'd be great to dlclose() this library at some point.
  void* lib = dlopen("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
  if (!lib) {
    throw ApiUnavailable{std::string(symbol) + " " + dlerror()};
  }
  return lib;
}

void* GetOpenCL(const char* symbol) {
  static void* lib = LoadOpenCL(symbol);
  return lib;
}

template <typename F>
F GetImpl(const char* symbol) {
  void* impl = dlsym(GetOpenCL(symbol), symbol);
  if (!impl) {
    throw ApiUnavailable{std::string(symbol) + " " + dlerror()};
  }
  return reinterpret_cast<F>(impl);
}

}  // namespace

#endif

cl_int BuildProgram(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options,
                    void(CL_CALLBACK* pfn_notify)(cl_program, void*), void* user_data) {
#ifdef OCL_STATIC_1_0

  return clBuildProgram(program, num_devices, device_list, options, pfn_notify, user_data);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_program, cl_uint, const cl_device_id*, const char*,
                                         void(CL_CALLBACK*)(cl_program, void*),  // NOLINT(readability/casting)
                                         void*)>("clBuildProgram");

  return impl(program, num_devices, device_list, options, pfn_notify, user_data);

#endif
}

cl_mem CreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret) {
#ifdef OCL_STATIC_1_0

  return clCreateBuffer(context, flags, size, host_ptr, errcode_ret);

#else

  static auto* impl = GetImpl<cl_mem (*)(cl_context, cl_mem_flags, size_t, void*, cl_int*)>("clCreateBuffer");

  return impl(context, flags, size, host_ptr, errcode_ret);

#endif
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties,
                                    cl_int* errcode_ret) {
#ifdef OCL_STATIC_1_0

  return clCreateCommandQueue(context, device, properties, errcode_ret);

#else

  static auto* impl = GetImpl<cl_command_queue (*)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*)>(
      "clCreateCommandQueue");

  return impl(context, device, properties, errcode_ret);

#endif
}

cl_context CreateContext(const cl_context_properties* properties, cl_uint num_devices, const cl_device_id* devices,
                         void(CL_CALLBACK* pfn_notify)(const char*, const void*, size_t, void*), void* user_data,
                         cl_int* errcode_ret) {
#ifdef OCL_STATIC_1_0

  return clCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);

#else

  static auto* impl = GetImpl<cl_context (*)(  // NOLINT(whitespace/parens)
      const cl_context_properties*, cl_uint, const cl_device_id*,
      void(CL_CALLBACK*)(const char*, const void*, size_t, void*),  // NOLINT(readability/casting)
      void*, cl_int*)>("clCreateContext");

  return impl(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);

#endif
}

cl_kernel CreateKernel(cl_program program, const char* kernel_name, cl_int* errcode_ret) {
#ifdef OCL_STATIC_1_0

  return clCreateKernel(program, kernel_name, errcode_ret);

#else

  static auto* impl = GetImpl<cl_kernel (*)(cl_program, const char*, cl_int*)>("clCreateKernel");

  return impl(program, kernel_name, errcode_ret);

#endif
}

cl_program CreateProgramWithSource(cl_context context, cl_uint count, const char** strings, const size_t* lengths,
                                   cl_int* errcode_ret) {
#ifdef OCL_STATIC_1_0

  return clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);

#else

  static auto* impl =
      GetImpl<cl_program (*)(cl_context, cl_uint, const char**, const size_t*, cl_int*)>("clCreateProgramWithSource");

  return impl(context, count, strings, lengths, errcode_ret);

#endif
}

cl_mem CreateSubBuffer(cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type buffer_create_type,
                       const void* buffer_create_info, cl_int* errcode_ret) {
#ifdef OCL_STATIC_1_1

  return clCreateSubBuffer(buffer, flags, buffer_create_type, buffer_create_info, errcode_ret);

#else

  static auto* impl =
      GetImpl<cl_mem (*)(cl_mem, cl_mem_flags, cl_buffer_create_type, const void*, cl_int*)>("clCreateSubBuffer");

  return impl(buffer, flags, buffer_create_type, buffer_create_info, errcode_ret);

#endif
}

cl_event CreateUserEvent(cl_context context, cl_int* errcode_ret) {
#ifdef OCL_STATIC_1_1

  return clCreateUserEvent(context, errcode_ret);

#else

  static auto* impl = GetImpl<cl_event (*)(cl_context, cl_int*)>("clCreateUserEvent");

  return impl(context, errcode_ret);

#endif
}

cl_int EnqueueCopyBuffer(cl_command_queue queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset,
                         size_t dst_offset, size_t size, cl_uint num_events_in_wait_list,
                         const cl_event* event_wait_list, cl_event* event) {
#ifdef OCL_STATIC_1_0

  return clEnqueueCopyBuffer(queue, src_buffer, dst_buffer, src_offset, dst_offset, size, num_events_in_wait_list,
                             event_wait_list, event);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint,
                                         const cl_event*, cl_event*)>("clEnqueueCopyBuffer");

  return impl(queue, src_buffer, dst_buffer, src_offset, dst_offset, size, num_events_in_wait_list, event_wait_list,
              event);

#endif
}

cl_int EnqueueFillBuffer(cl_command_queue queue, cl_mem buffer, const void* pattern, size_t pattern_size, size_t offset,
                         size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                         cl_event* event) {
#ifdef OCL_STATIC_1_2

  return clEnqueueFillBuffer(queue, buffer, pattern, pattern_size, offset, size, num_events_in_wait_list,
                             event_wait_list, event);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_command_queue, cl_mem, const void*, size_t, size_t, size_t, cl_uint,
                                         const cl_event*, cl_event*)>("clEnqueueFillBuffer");

  return impl(queue, buffer, pattern, pattern_size, offset, size, num_events_in_wait_list, event_wait_list, event);

#endif
}

void* EnqueueMapBuffer(cl_command_queue queue, cl_mem buffer, cl_bool blocking_map, cl_map_flags map_flags,
                       size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                       cl_event* event, cl_int* errcode_ret) {
#ifdef OCL_STATIC_1_0

  return clEnqueueMapBuffer(queue, buffer, blocking_map, map_flags, offset, size, num_events_in_wait_list,
                            event_wait_list, event, errcode_ret);

#else

  static auto* impl = GetImpl<void* (*)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint,
                                        const cl_event*, cl_event*, cl_int*)>("clEnqueueMapBuffer");

  return impl(queue, buffer, blocking_map, map_flags, offset, size, num_events_in_wait_list, event_wait_list, event,
              errcode_ret);

#endif
}

cl_int EnqueueMarkerWithWaitList(cl_command_queue queue, cl_uint num_events_in_wait_list,
                                 const cl_event* event_wait_list, cl_event* event) {
#ifdef OCL_STATIC_1_2

  return clEnqueueMarkerWithWaitList(queue, num_events_in_wait_list, event_wait_list, event);

#else

  static auto* impl =
      GetImpl<cl_int (*)(cl_command_queue, cl_uint, const cl_event*, cl_event*)>("clEnqueueMarkerWithWaitList");

  return impl(queue, num_events_in_wait_list, event_wait_list, event);

#endif
}

cl_int EnqueueNDRangeKernel(cl_command_queue queue, cl_kernel kernel, cl_uint work_dim,
                            const size_t* global_work_offset, const size_t* global_work_size,
                            const size_t* local_work_size, cl_uint num_events_in_wait_list,
                            const cl_event* event_wait_list, cl_event* event) {
#ifdef OCL_STATIC_1_0

  return clEnqueueNDRangeKernel(queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size,
                                num_events_in_wait_list, event_wait_list, event);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*,
                                         const size_t*, cl_uint, const cl_event*, cl_event*)>("clEnqueueNDRangeKernel");

  return impl(queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list,
              event_wait_list, event);

#endif
}

cl_int EnqueueReadBuffer(cl_command_queue queue, cl_mem buffer, bool blocking_read, size_t offset, size_t size,
                         void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
#ifdef OCL_STATIC_1_0

  return clEnqueueReadBuffer(queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list,
                             event);

#else

  static auto* impl =
      GetImpl<cl_int (*)(cl_command_queue, cl_mem, bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*)>(
          "clEnqueueReadBuffer");

  return impl(queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);

#endif
}

cl_int EnqueueSVMMemFill(cl_command_queue queue, void* svm_ptr, const void* pattern, size_t pattern_size, size_t size,
                         cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
#ifdef OCL_STATIC_2_0

  return clEnqueueSVMMemFill(queue, svm_ptr, pattern, pattern_size, size, num_events_in_wait_list, event_wait_list,
                             event);

#else

  static auto* impl =
      GetImpl<cl_int (*)(cl_command_queue, void*, const void*, size_t, size_t, cl_uint, const cl_event*, cl_event*)>(
          "clEnqueueSVMMemFill");

  return impl(queue, svm_ptr, pattern, pattern_size, size, num_events_in_wait_list, event_wait_list, event);

#endif
}

cl_int EnqueueUnmapMemObject(cl_command_queue queue, cl_mem memobj, void* mapped_ptr, cl_uint num_events_in_wait_list,
                             const cl_event* event_wait_list, cl_event* event) {
#ifdef OCL_STATIC_1_0

  return clEnqueueUnmapMemObject(queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);

#else
  static auto* impl = GetImpl<cl_int (*)(cl_command_queue, cl_mem, void*, cl_uint, const cl_event*, cl_event*)>(
      "clEnqueueUnmapMemObject");

  return impl(queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);

#endif
}

cl_int EnqueueWriteBuffer(cl_command_queue queue, cl_mem buffer, bool blocking_write, size_t offset, size_t size,
                          const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                          cl_event* event) {
#ifdef OCL_STATIC_1_0

  return clEnqueueWriteBuffer(queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list,
                              event_wait_list, event);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_command_queue, cl_mem, bool, size_t, size_t, const void*, cl_uint,
                                         const cl_event*, cl_event*)>("clEnqueueWriteBuffer");

  return impl(queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);

#endif
}

cl_int Finish(cl_command_queue queue) {
#ifdef OCL_STATIC_1_0

  return clFinish(queue);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_command_queue)>("clFinish");

  return impl(queue);

#endif
}

cl_int Flush(cl_command_queue queue) {
#ifdef OCL_STATIC_1_0

  return clFlush(queue);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_command_queue)>("clFlush");

  return impl(queue);

#endif
}

cl_int GetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id* devices,
                    cl_uint* num_devices) {
#ifdef OCL_STATIC_1_0

  return clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices);

#else

  static auto* impl =
      GetImpl<cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*)>("clGetDeviceIDs");

  return impl(platform, device_type, num_entries, devices, num_devices);

#endif
}

cl_int GetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void* param_value,
                     size_t* param_value_size_ret) {
#ifdef OCL_STATIC_1_0

  return clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_device_id, cl_device_info, size_t, void*, size_t*)>("clGetDeviceInfo");

  return impl(device, param_name, param_value_size, param_value, param_value_size_ret);

#endif
}

cl_int GetEventInfo(cl_event event, cl_event_info param_name, size_t param_value_size, void* param_value,
                    size_t* param_value_size_ret) {
#ifdef OCL_STATIC_1_0

  return clGetEventInfo(event, param_name, param_value_size, param_value, param_value_size_ret);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_event, cl_event_info, size_t, void*, size_t*)>("clGetEventInfo");

  return impl(event, param_name, param_value_size, param_value, param_value_size_ret);

#endif
}

cl_int GetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void* param_value,
                             size_t* param_value_size_ret) {
#ifdef OCL_STATIC_1_0

  return clGetEventProfilingInfo(event, param_name, param_value_size, param_value, param_value_size_ret);

#else

  static auto* impl =
      GetImpl<cl_int (*)(cl_event, cl_profiling_info, size_t, void*, size_t*)>("clGetEventProfilingInfo");

  return impl(event, param_name, param_value_size, param_value, param_value_size_ret);

#endif
}

cl_int GetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name,
                              size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
#ifdef OCL_STATIC_1_0

  return clGetKernelWorkGroupInfo(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*, size_t*)>(
      "clGetKernelWorkGroupInfo");

  return impl(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);

#endif
}

cl_int GetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms) {
#ifdef OCL_STATIC_1_0

  return clGetPlatformIDs(num_entries, platforms, num_platforms);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_uint, cl_platform_id*, cl_uint*)>("clGetPlatformIDs");

  return impl(num_entries, platforms, num_platforms);

#endif
}

cl_int GetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void* param_value,
                       size_t* param_value_size_ret) {
#ifdef OCL_STATIC_1_0

  return clGetPlatformInfo(platform, param_name, param_value_size, param_value, param_value_size_ret);

#else

  static auto* impl =
      GetImpl<cl_int (*)(cl_platform_id, cl_platform_info, size_t, void*, size_t*)>("clGetPlatformInfo");

  return impl(platform, param_name, param_value_size, param_value, param_value_size_ret);

#endif
}

cl_int GetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name,
                           size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
#ifdef OCL_STATIC_1_0

  return clGetProgramBuildInfo(program, device, param_name, param_value_size, param_value, param_value_size_ret);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*)>(
      "clGetProgramBuildInfo");

  return impl(program, device, param_name, param_value_size, param_value, param_value_size_ret);

#endif
}

cl_int GetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void* param_value,
                      size_t* param_value_size_ret) {
#ifdef OCL_STATIC_1_0

  return clGetProgramInfo(program, param_name, param_value_size, param_value, param_value_size_ret);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_program, cl_program_info, size_t, void*, size_t*)>("clGetProgramInfo");

  return impl(program, param_name, param_value_size, param_value, param_value_size_ret);

#endif
}

cl_int ReleaseContext(cl_context c) {
#ifdef OCL_STATIC_1_0

  return clReleaseContext(c);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_context)>("clReleaseContext");
  return impl(c);

#endif
}

cl_int ReleaseCommandQueue(cl_command_queue c) {
#ifdef OCL_STATIC_1_0

  return clReleaseCommandQueue(c);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_command_queue)>("clReleaseCommandQueue");
  return impl(c);

#endif
}

cl_int ReleaseEvent(cl_event e) {
#ifdef OCL_STATIC_1_0

  return clReleaseEvent(e);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_event)>("clReleaseEvent");
  return impl(e);

#endif
}

cl_int ReleaseKernel(cl_kernel k) {
#ifdef OCL_STATIC_1_0

  return clReleaseKernel(k);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_kernel)>("clReleaseKernel");
  return impl(k);

#endif
}

cl_int ReleaseMemObject(cl_mem m) {
#ifdef OCL_STATIC_1_0

  return clReleaseMemObject(m);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_mem)>("clReleaseMemObject");
  return impl(m);

#endif
}

cl_int ReleaseProgram(cl_program p) {
#ifdef OCL_STATIC_1_0

  return clReleaseProgram(p);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_program)>("clReleaseProgram");
  return impl(p);

#endif
}

cl_int RetainContext(cl_context c) {
#ifdef OCL_STATIC_1_0

  return clRetainContext(c);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_context)>("clRetainContext");
  return impl(c);

#endif
}

cl_int RetainCommandQueue(cl_command_queue c) {
#ifdef OCL_STATIC_1_0

  return clRetainCommandQueue(c);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_command_queue)>("clRetainCommandQueue");
  return impl(c);

#endif
}

cl_int RetainEvent(cl_event e) {
#ifdef OCL_STATIC_1_0

  return clRetainEvent(e);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_event)>("clRetainEvent");
  return impl(e);

#endif
}

cl_int RetainKernel(cl_kernel k) {
#ifdef OCL_STATIC_1_0

  return clRetainKernel(k);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_kernel)>("clRetainKernel");
  return impl(k);

#endif
}

cl_int RetainMemObject(cl_mem m) {
#ifdef OCL_STATIC_1_0

  return clRetainMemObject(m);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_mem)>("clRetainMemObject");
  return impl(m);

#endif
}

cl_int RetainProgram(cl_program p) {
#ifdef OCL_STATIC_1_0

  return clRetainProgram(p);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_program)>("clRetainProgram");
  return impl(p);

#endif
}

cl_int SetEventCallback(cl_event event, cl_int command_exec_callback_type,
                        void(CL_CALLBACK* pfn_notify)(cl_event, cl_int, void*), void* user_data) {
#ifdef OCL_STATIC_1_1

  return clSetEventCallback(event, command_exec_callback_type, pfn_notify, user_data);

#else

  static auto* impl =
      GetImpl<cl_int (*)(cl_event, cl_int, void(CL_CALLBACK*)(cl_event, cl_int, void*),  // NOLINT(readability/casting)
                         void*)>("clSetEventCallback");
  return impl(event, command_exec_callback_type, pfn_notify, user_data);

#endif
}

cl_int SetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value) {
#ifdef OCL_STATIC_1_0

  return clSetKernelArg(kernel, arg_index, arg_size, arg_value);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_kernel, cl_uint, size_t, const void*)>("clSetKernelArg");
  return impl(kernel, arg_index, arg_size, arg_value);

#endif
}

cl_int SetKernelArgSVMPointer(cl_kernel kernel, cl_uint arg_index, const void* arg_value) {
#ifdef OCL_STATIC_2_0

  return clSetKernelArgSVMPointer(kernel, arg_index, arg_value);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_kernel, cl_uint, const void*)>("clSetKernelArgSVMPointer");
  return impl(kernel, arg_index, arg_value);

#endif
}

cl_int SetUserEventStatus(cl_event event, cl_int execution_status) {
#ifdef OCL_STATIC_1_1

  return clSetUserEventStatus(event, execution_status);

#else

  static auto* impl = GetImpl<cl_int (*)(cl_event, cl_int)>("clSetUserEventStatus");

  return impl(event, execution_status);

#endif
}

void* SVMAlloc(cl_context context, cl_svm_mem_flags flags, size_t size, cl_uint alignment) {
#ifdef OCL_STATIC_2_0

  return clSVMAlloc(context, flags, size, alignment);

#else

  static auto* impl = GetImpl<void* (*)(cl_context, cl_svm_mem_flags, size_t, cl_uint)>("clSVMAlloc");
  return impl(context, flags, size, alignment);

#endif
}

void SVMFree(cl_context context, void* svm_pointer) {
#ifdef OCL_STATIC_2_0

  return clSVMFree(context, svm_pointer);

#else

  static auto* impl = GetImpl<void (*)(cl_context, void*)>("clSVMFree");
  impl(context, svm_pointer);

#endif
}

}  // namespace opencl
}  // namespace shim
}  // namespace vertexai
