// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include <boost/regex.hpp>

#include "tile/hal/opencl/ocl.h"
#include "tile/hal/opencl/opencl.pb.h"
#include "tile/proto/hal.pb.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

// A few wrappers to simplify retrieving OpenCL platform and device information.
template <cl_device_info Param, typename T>
struct CLDeviceInfo {
  static T Read(cl_device_id did) {
    T value = 0;
    Err err = clGetDeviceInfo(did, Param, sizeof(value), reinterpret_cast<char*>(&value), nullptr);
    if (err && err.code() != CL_INVALID_VALUE) {
      Err::Check(err, "reading OpenCL device info");
    }
    return value;
  }
};

template <cl_platform_info Param, typename T>
struct CLPlatformInfo {
  static T Read(cl_platform_id pid) {
    T value = 0;
    Err err = clGetPlatformInfo(pid, Param, sizeof(value), reinterpret_cast<char*>(&value), nullptr);
    if (err && err.code() != CL_INVALID_VALUE && err.code() != CL_INVALID_DEVICE) {
      Err::Check(err, "reading OpenCL platform info");
    }
    return value;
  }
};

template <cl_device_info Param>
struct CLDeviceInfo<Param, std::string> {
  static std::string Read(cl_device_id did) {
    std::size_t value_size;
    Err err = clGetDeviceInfo(did, Param, 0, nullptr, &value_size);
    if (err) {
      if (err.code() != CL_INVALID_VALUE) {
        Err::Check(err, "reading OpenCL device info size");
      }
      return std::string();
    }
    std::string value(value_size, '\0');
    Err::Check(clGetDeviceInfo(did, Param, value.size(), const_cast<char*>(value.c_str()), nullptr),
               "reading OpenCL device info char[] data");
    if (value.size() && value[value.size() - 1] == '\0') {
      value.pop_back();
    }
    return value;
  }
};

template <cl_platform_info Param>
struct CLPlatformInfo<Param, std::string> {
  static std::string Read(cl_platform_id pid) {
    std::size_t value_size;
    Err err = clGetPlatformInfo(pid, Param, 0, nullptr, &value_size);
    if (err) {
      if (err.code() != CL_INVALID_VALUE) {
        Err::Check(err, "reading OpenCL platform info size");
      }
      return std::string();
    }
    std::string value(value_size, '\0');
    Err::Check(clGetPlatformInfo(pid, Param, value.size(), const_cast<char*>(value.c_str()), nullptr),
               "reading OpenCL platform info char[] data");
    if (value.size() && value[value.size() - 1] == '\0') {
      value.pop_back();
    }
    return value;
  }
};

template <cl_device_info Param, typename T>
struct CLDeviceInfo<Param, std::vector<T>> {
  static std::vector<T> Read(cl_device_id did) {
    std::size_t value_size;
    Err err = clGetDeviceInfo(did, Param, 0, nullptr, &value_size);
    if (err) {
      if (err.code() != CL_INVALID_VALUE) {
        Err::Check(err, "reading OpenCL device info size");
      }
      return std::vector<T>();
    }
    std::vector<T> value(value_size / sizeof(T));
    Err::Check(clGetDeviceInfo(did, Param, value.size() * sizeof(T), value.data(), nullptr),
               "reading OpenCL device info array data");
    return value;
  }
};

// Specialize to associate IDs to their corresponding type information.

template <cl_platform_info Param>
struct CLInfoType;

template <>
struct CLInfoType<CL_PLATFORM_PROFILE> : CLPlatformInfo<CL_PLATFORM_PROFILE, std::string> {};
template <>
struct CLInfoType<CL_PLATFORM_VERSION> : CLPlatformInfo<CL_PLATFORM_VERSION, std::string> {};
template <>
struct CLInfoType<CL_PLATFORM_NAME> : CLPlatformInfo<CL_PLATFORM_NAME, std::string> {};
template <>
struct CLInfoType<CL_PLATFORM_VENDOR> : CLPlatformInfo<CL_PLATFORM_VENDOR, std::string> {};
template <>
struct CLInfoType<CL_PLATFORM_EXTENSIONS> : CLPlatformInfo<CL_PLATFORM_EXTENSIONS, std::string> {};
#ifndef __APPLE__
template <>
struct CLInfoType<CL_PLATFORM_HOST_TIMER_RESOLUTION> : CLPlatformInfo<CL_PLATFORM_HOST_TIMER_RESOLUTION, cl_ulong> {};
#endif

template <>
struct CLInfoType<CL_DEVICE_TYPE> : CLDeviceInfo<CL_DEVICE_TYPE, cl_device_type> {};
template <>
struct CLInfoType<CL_DEVICE_VENDOR_ID> : CLDeviceInfo<CL_DEVICE_VENDOR_ID, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_MAX_COMPUTE_UNITS> : CLDeviceInfo<CL_DEVICE_MAX_COMPUTE_UNITS, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS> : CLDeviceInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_MAX_WORK_GROUP_SIZE> : CLDeviceInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE, size_t> {};
template <>
struct CLInfoType<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR>
    : CLDeviceInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT>
    : CLDeviceInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT> : CLDeviceInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, cl_uint> {
};
template <>
struct CLInfoType<CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG>
    : CLDeviceInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>
    : CLDeviceInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>
    : CLDeviceInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF>
    : CLDeviceInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR> : CLDeviceInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT> : CLDeviceInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_NATIVE_VECTOR_WIDTH_INT> : CLDeviceInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG> : CLDeviceInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT> : CLDeviceInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE> : CLDeviceInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, cl_uint> {
};
template <>
struct CLInfoType<CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF> : CLDeviceInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_MAX_CLOCK_FREQUENCY> : CLDeviceInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_ADDRESS_BITS> : CLDeviceInfo<CL_DEVICE_ADDRESS_BITS, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_MAX_MEM_ALLOC_SIZE> : CLDeviceInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE, cl_ulong> {};
template <>
struct CLInfoType<CL_DEVICE_IMAGE_SUPPORT> : CLDeviceInfo<CL_DEVICE_IMAGE_SUPPORT, cl_bool> {};
#ifndef __APPLE__
template <>
struct CLInfoType<CL_DEVICE_MAX_READ_IMAGE_ARGS> : CLDeviceInfo<CL_DEVICE_MAX_READ_IMAGE_ARGS, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_MAX_WRITE_IMAGE_ARGS> : CLDeviceInfo<CL_DEVICE_MAX_WRITE_IMAGE_ARGS, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS> : CLDeviceInfo<CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_IL_VERSION> : CLDeviceInfo<CL_DEVICE_IL_VERSION, std::string> {};
#endif
template <>
struct CLInfoType<CL_DEVICE_IMAGE2D_MAX_WIDTH> : CLDeviceInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH, size_t> {};
template <>
struct CLInfoType<CL_DEVICE_IMAGE2D_MAX_HEIGHT> : CLDeviceInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT, size_t> {};
template <>
struct CLInfoType<CL_DEVICE_IMAGE3D_MAX_WIDTH> : CLDeviceInfo<CL_DEVICE_IMAGE3D_MAX_WIDTH, size_t> {};
template <>
struct CLInfoType<CL_DEVICE_IMAGE3D_MAX_HEIGHT> : CLDeviceInfo<CL_DEVICE_IMAGE3D_MAX_HEIGHT, size_t> {};
template <>
struct CLInfoType<CL_DEVICE_IMAGE3D_MAX_DEPTH> : CLDeviceInfo<CL_DEVICE_IMAGE3D_MAX_DEPTH, size_t> {};
template <>
struct CLInfoType<CL_DEVICE_IMAGE_MAX_BUFFER_SIZE> : CLDeviceInfo<CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, size_t> {};
template <>
struct CLInfoType<CL_DEVICE_IMAGE_MAX_ARRAY_SIZE> : CLDeviceInfo<CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, size_t> {};
template <>
struct CLInfoType<CL_DEVICE_MAX_SAMPLERS> : CLDeviceInfo<CL_DEVICE_MAX_SAMPLERS, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_IMAGE_PITCH_ALIGNMENT> : CLDeviceInfo<CL_DEVICE_IMAGE_PITCH_ALIGNMENT, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT>
    : CLDeviceInfo<CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT, cl_uint> {};
#ifndef __APPLE__
template <>
struct CLInfoType<CL_DEVICE_MAX_PIPE_ARGS> : CLDeviceInfo<CL_DEVICE_MAX_PIPE_ARGS, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS>
    : CLDeviceInfo<CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS, cl_uint> {};
#endif
template <>
struct CLInfoType<CL_DEVICE_MAX_PARAMETER_SIZE> : CLDeviceInfo<CL_DEVICE_MAX_PARAMETER_SIZE, size_t> {};
template <>
struct CLInfoType<CL_DEVICE_MEM_BASE_ADDR_ALIGN> : CLDeviceInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_SINGLE_FP_CONFIG> : CLDeviceInfo<CL_DEVICE_SINGLE_FP_CONFIG, cl_device_fp_config> {};
template <>
struct CLInfoType<CL_DEVICE_DOUBLE_FP_CONFIG> : CLDeviceInfo<CL_DEVICE_DOUBLE_FP_CONFIG, cl_device_fp_config> {};
template <>
struct CLInfoType<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>
    : CLDeviceInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, cl_device_mem_cache_type> {};
template <>
struct CLInfoType<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE> : CLDeviceInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE> : CLDeviceInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cl_ulong> {};
template <>
struct CLInfoType<CL_DEVICE_GLOBAL_MEM_SIZE> : CLDeviceInfo<CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong> {};
template <>
struct CLInfoType<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE> : CLDeviceInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, cl_ulong> {};
template <>
struct CLInfoType<CL_DEVICE_MAX_CONSTANT_ARGS> : CLDeviceInfo<CL_DEVICE_MAX_CONSTANT_ARGS, cl_uint> {};
#ifndef __APPLE__
template <>
struct CLInfoType<CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE> : CLDeviceInfo<CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, size_t> {};
template <>
struct CLInfoType<CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE>
    : CLDeviceInfo<CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE, size_t> {};
#endif
template <>
struct CLInfoType<CL_DEVICE_HOST_UNIFIED_MEMORY> : CLDeviceInfo<CL_DEVICE_HOST_UNIFIED_MEMORY, cl_bool> {};
template <>
struct CLInfoType<CL_DEVICE_LOCAL_MEM_TYPE> : CLDeviceInfo<CL_DEVICE_LOCAL_MEM_TYPE, cl_device_local_mem_type> {};
template <>
struct CLInfoType<CL_DEVICE_LOCAL_MEM_SIZE> : CLDeviceInfo<CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong> {};
template <>
struct CLInfoType<CL_DEVICE_ERROR_CORRECTION_SUPPORT> : CLDeviceInfo<CL_DEVICE_ERROR_CORRECTION_SUPPORT, cl_bool> {};
template <>
struct CLInfoType<CL_DEVICE_PROFILING_TIMER_RESOLUTION> : CLDeviceInfo<CL_DEVICE_PROFILING_TIMER_RESOLUTION, size_t> {};
template <>
struct CLInfoType<CL_DEVICE_ENDIAN_LITTLE> : CLDeviceInfo<CL_DEVICE_ENDIAN_LITTLE, cl_bool> {};
template <>
struct CLInfoType<CL_DEVICE_AVAILABLE> : CLDeviceInfo<CL_DEVICE_AVAILABLE, cl_bool> {};
template <>
struct CLInfoType<CL_DEVICE_COMPILER_AVAILABLE> : CLDeviceInfo<CL_DEVICE_COMPILER_AVAILABLE, cl_bool> {};
template <>
struct CLInfoType<CL_DEVICE_LINKER_AVAILABLE> : CLDeviceInfo<CL_DEVICE_LINKER_AVAILABLE, cl_bool> {};
template <>
struct CLInfoType<CL_DEVICE_EXECUTION_CAPABILITIES>
    : CLDeviceInfo<CL_DEVICE_EXECUTION_CAPABILITIES, cl_device_exec_capabilities> {};
#ifdef __APPLE__
template <>
struct CLInfoType<CL_DEVICE_QUEUE_PROPERTIES> : CLDeviceInfo<CL_DEVICE_QUEUE_PROPERTIES, cl_command_queue_properties> {
};
#else
template <>
struct CLInfoType<CL_DEVICE_QUEUE_ON_HOST_PROPERTIES>
    : CLDeviceInfo<CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, cl_command_queue_properties> {};
template <>
struct CLInfoType<CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES>
    : CLDeviceInfo<CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES, cl_command_queue_properties> {};
template <>
struct CLInfoType<CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE>
    : CLDeviceInfo<CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE> : CLDeviceInfo<CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_MAX_ON_DEVICE_QUEUES> : CLDeviceInfo<CL_DEVICE_MAX_ON_DEVICE_QUEUES, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_MAX_ON_DEVICE_EVENTS> : CLDeviceInfo<CL_DEVICE_MAX_ON_DEVICE_EVENTS, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_BUILT_IN_KERNELS> : CLDeviceInfo<CL_DEVICE_BUILT_IN_KERNELS, std::string> {};
#endif
template <>
struct CLInfoType<CL_DEVICE_NAME> : CLDeviceInfo<CL_DEVICE_NAME, std::string> {};
template <>
struct CLInfoType<CL_DEVICE_VENDOR> : CLDeviceInfo<CL_DEVICE_VENDOR, std::string> {};
template <>
struct CLInfoType<CL_DRIVER_VERSION> : CLDeviceInfo<CL_DRIVER_VERSION, std::string> {};
template <>
struct CLInfoType<CL_DEVICE_PROFILE> : CLDeviceInfo<CL_DEVICE_PROFILE, std::string> {};
template <>
struct CLInfoType<CL_DEVICE_VERSION> : CLDeviceInfo<CL_DEVICE_VERSION, std::string> {};
template <>
struct CLInfoType<CL_DEVICE_OPENCL_C_VERSION> : CLDeviceInfo<CL_DEVICE_OPENCL_C_VERSION, std::string> {};
template <>
struct CLInfoType<CL_DEVICE_EXTENSIONS> : CLDeviceInfo<CL_DEVICE_EXTENSIONS, std::string> {};
template <>
struct CLInfoType<CL_DEVICE_PRINTF_BUFFER_SIZE> : CLDeviceInfo<CL_DEVICE_PRINTF_BUFFER_SIZE, size_t> {};
template <>
struct CLInfoType<CL_DEVICE_PREFERRED_INTEROP_USER_SYNC>
    : CLDeviceInfo<CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, cl_bool> {};
template <>
struct CLInfoType<CL_DEVICE_PARTITION_MAX_SUB_DEVICES> : CLDeviceInfo<CL_DEVICE_PARTITION_MAX_SUB_DEVICES, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_PARTITION_PROPERTIES>
    : CLDeviceInfo<CL_DEVICE_PARTITION_PROPERTIES, std::vector<cl_device_partition_property>> {};
template <>
struct CLInfoType<CL_DEVICE_PARTITION_AFFINITY_DOMAIN>
    : CLDeviceInfo<CL_DEVICE_PARTITION_AFFINITY_DOMAIN, cl_device_affinity_domain> {};
template <>
struct CLInfoType<CL_DEVICE_PARTITION_TYPE>
    : CLDeviceInfo<CL_DEVICE_PARTITION_TYPE, std::vector<cl_device_affinity_domain>> {};
#ifndef __APPLE__
template <>
struct CLInfoType<CL_DEVICE_SVM_CAPABILITIES> : CLDeviceInfo<CL_DEVICE_SVM_CAPABILITIES, cl_device_svm_capabilities> {};
template <>
struct CLInfoType<CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT>
    : CLDeviceInfo<CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT>
    : CLDeviceInfo<CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT>
    : CLDeviceInfo<CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_MAX_NUM_SUB_GROUPS> : CLDeviceInfo<CL_DEVICE_MAX_NUM_SUB_GROUPS, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS>
    : CLDeviceInfo<CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS, cl_bool> {};
#endif
#ifdef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
template <>
struct CLInfoType<CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV>
    : CLDeviceInfo<CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV>
    : CLDeviceInfo<CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_REGISTERS_PER_BLOCK_NV> : CLDeviceInfo<CL_DEVICE_REGISTERS_PER_BLOCK_NV, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_WARP_SIZE_NV> : CLDeviceInfo<CL_DEVICE_WARP_SIZE_NV, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_GPU_OVERLAP_NV> : CLDeviceInfo<CL_DEVICE_GPU_OVERLAP_NV, cl_bool> {};
template <>
struct CLInfoType<CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV> : CLDeviceInfo<CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, cl_bool> {};
template <>
struct CLInfoType<CL_DEVICE_INTEGRATED_MEMORY_NV> : CLDeviceInfo<CL_DEVICE_INTEGRATED_MEMORY_NV, cl_bool> {};
#endif
#ifdef CL_DEVICE_BOARD_NAME_AMD
template <>
struct CLInfoType<CL_DEVICE_BOARD_NAME_AMD> : CLDeviceInfo<CL_DEVICE_BOARD_NAME_AMD, std::string> {};
template <>
struct CLInfoType<CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD> : CLDeviceInfo<CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD> : CLDeviceInfo<CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD, cl_uint> {
};
template <>
struct CLInfoType<CL_DEVICE_WAVEFRONT_WIDTH_AMD> : CLDeviceInfo<CL_DEVICE_WAVEFRONT_WIDTH_AMD, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD> : CLDeviceInfo<CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD>
    : CLDeviceInfo<CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD>
    : CLDeviceInfo<CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD>
    : CLDeviceInfo<CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD, cl_uint> {};
template <>
struct CLInfoType<CL_DEVICE_LOCAL_MEM_BANKS_AMD> : CLDeviceInfo<CL_DEVICE_LOCAL_MEM_BANKS_AMD, cl_uint> {};
#endif

// Define a function to read the CL information.

template <cl_platform_info P, typename T>
auto CLInfo(T id) {
  return CLInfoType<P>::Read(id);
}

// OpenCL has a number of platform/device info bits that are strings containing
// a whitespace-separated list of values.  This template slightly automates the
// process of splitting those strings apart.
template <class T>
void ForEachElt(const std::string& elts_str, T t, boost::regex elt_re = boost::regex{R"(\S+)"}) {
  std::for_each(boost::sregex_iterator{elts_str.begin(), elts_str.end(), elt_re}, boost::sregex_iterator(),
                [&t](const boost::smatch& match) {
                  std::string elt = match.str();
                  t(std::move(elt));
                });
}

hal::proto::HardwareInfo GetHardwareInfo(const proto::DeviceInfo& info);

void LogInfo(const std::string& prefix, const google::protobuf::Message& info);

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
