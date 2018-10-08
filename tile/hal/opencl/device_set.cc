// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/opencl/device_set.h"

#include <string>
#include <utility>

#include <boost/regex.hpp>

#include "base/util/compat.h"
#include "base/util/error.h"
#include "tile/hal/opencl/info.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {
namespace {

void OnErr(const char* errinfo, const void* priv_info, size_t cb, void* user_data) {
  std::string err{errinfo};
  IVLOG(1, "OpenCL: " << err << " (cb=" << cb << ")");
}

proto::PlatformInfo GetPlatformInfo(cl_platform_id pid) {
  proto::PlatformInfo info;

  info.set_profile(CLInfo<CL_PLATFORM_PROFILE>(pid));
  info.set_version(CLInfo<CL_PLATFORM_VERSION>(pid));
  info.set_name(CLInfo<CL_PLATFORM_NAME>(pid));
  info.set_vendor(CLInfo<CL_PLATFORM_VENDOR>(pid));
  ForEachElt(CLInfo<CL_PLATFORM_EXTENSIONS>(pid), [&info](std::string ext) { info.add_extension(ext); });
#ifndef __APPLE__
  info.set_host_timer_resolution_ns(CLInfo<CL_PLATFORM_HOST_TIMER_RESOLUTION>(pid));
#endif

  return info;
}

proto::DeviceInfo GetDeviceInfo(cl_device_id did, std::uint32_t pidx, const proto::PlatformInfo& pinfo) {
  proto::DeviceInfo info;

  switch (CLInfo<CL_DEVICE_TYPE>(did)) {
    case CL_DEVICE_TYPE_CPU:
      info.set_type(hal::proto::HardwareType::CPU);
      break;
    case CL_DEVICE_TYPE_GPU:
      info.set_type(hal::proto::HardwareType::GPU);
      break;
    case CL_DEVICE_TYPE_ACCELERATOR:
      info.set_type(hal::proto::HardwareType::Accelerator);
      break;
    case CL_DEVICE_TYPE_CUSTOM:
      info.set_type(hal::proto::HardwareType::Custom);
      break;
  }
  info.set_vendor_id(CLInfo<CL_DEVICE_VENDOR_ID>(did));
  info.set_max_compute_units(CLInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(did));
  std::vector<size_t> sizes(CLInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>(did));
  Err::Check(clGetDeviceInfo(did, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * sizes.size(),
                             reinterpret_cast<char*>(sizes.data()), nullptr),
             "reading OpenCL device info");
  for (size_t size : sizes) {
    info.add_work_item_dimension_size(size);
  }
  info.set_max_work_group_size(CLInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(did));
  info.set_preferred_vector_width_char(CLInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR>(did));
  info.set_preferred_vector_width_short(CLInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT>(did));
  info.set_preferred_vector_width_int(CLInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>(did));
  info.set_preferred_vector_width_long(CLInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG>(did));
  info.set_preferred_vector_width_float(CLInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>(did));
  info.set_preferred_vector_width_double(CLInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>(did));
  info.set_preferred_vector_width_half(CLInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF>(did));
  info.set_native_vector_width_char(CLInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR>(did));
  info.set_native_vector_width_short(CLInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT>(did));
  info.set_native_vector_width_int(CLInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_INT>(did));
  info.set_native_vector_width_long(CLInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG>(did));
  info.set_native_vector_width_float(CLInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT>(did));
  info.set_native_vector_width_double(CLInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>(did));
  info.set_native_vector_width_half(CLInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF>(did));
  info.set_max_clock_frequency_mhz(CLInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>(did));
  info.set_address_bits(CLInfo<CL_DEVICE_ADDRESS_BITS>(did));
  info.set_max_mem_alloc_size(CLInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>(did));
  info.set_image_support(CLInfo<CL_DEVICE_IMAGE_SUPPORT>(did));
#ifndef __APPLE__
  info.set_max_read_image_args(CLInfo<CL_DEVICE_MAX_READ_IMAGE_ARGS>(did));
  info.set_max_write_image_args(CLInfo<CL_DEVICE_MAX_WRITE_IMAGE_ARGS>(did));
  info.set_max_read_write_image_args(CLInfo<CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS>(did));
  ForEachElt(CLInfo<CL_DEVICE_IL_VERSION>(did), [&info](std::string il) { info.add_il_version(il); });
#endif
  info.set_image2d_max_width(CLInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>(did));
  info.set_image2d_max_height(CLInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>(did));
  info.set_image3d_max_width(CLInfo<CL_DEVICE_IMAGE3D_MAX_WIDTH>(did));
  info.set_image3d_max_height(CLInfo<CL_DEVICE_IMAGE3D_MAX_HEIGHT>(did));
  info.set_image3d_max_depth(CLInfo<CL_DEVICE_IMAGE3D_MAX_DEPTH>(did));
  info.set_image_max_buffer_size(CLInfo<CL_DEVICE_IMAGE_MAX_BUFFER_SIZE>(did));
  info.set_image_max_array_size(CLInfo<CL_DEVICE_IMAGE_MAX_ARRAY_SIZE>(did));
  info.set_max_samplers(CLInfo<CL_DEVICE_MAX_SAMPLERS>(did));
  info.set_image_pitch_alignment(CLInfo<CL_DEVICE_IMAGE_PITCH_ALIGNMENT>(did));
  info.set_image_base_address_alignment(CLInfo<CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT>(did));
#ifndef __APPLE__
  info.set_max_pipe_args(CLInfo<CL_DEVICE_MAX_PIPE_ARGS>(did));
  info.set_pipe_max_active_reservations(CLInfo<CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS>(did));
#endif
  info.set_max_parameter_size(CLInfo<CL_DEVICE_MAX_PARAMETER_SIZE>(did));
  info.set_mem_base_addr_align(CLInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>(did));
  auto fpc = CLInfo<CL_DEVICE_SINGLE_FP_CONFIG>(did);
  if (fpc & CL_FP_DENORM) {
    info.add_single_fp_config(proto::DeviceFPConfig::Denorm);
  }
  if (fpc & CL_FP_INF_NAN) {
    info.add_single_fp_config(proto::DeviceFPConfig::InfNan);
  }
  if (fpc & CL_FP_ROUND_TO_NEAREST) {
    info.add_single_fp_config(proto::DeviceFPConfig::RoundToNearest);
  }
  if (fpc & CL_FP_ROUND_TO_ZERO) {
    info.add_single_fp_config(proto::DeviceFPConfig::RoundToZero);
  }
  if (fpc & CL_FP_ROUND_TO_INF) {
    info.add_single_fp_config(proto::DeviceFPConfig::RoundToInf);
  }
  if (fpc & CL_FP_FMA) {
    info.add_single_fp_config(proto::DeviceFPConfig::FusedMultiplyAdd);
  }
  if (fpc & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) {
    info.add_single_fp_config(proto::DeviceFPConfig::CorrectlyRoundedDivideSqrt);
  }
  if (fpc & CL_FP_SOFT_FLOAT) {
    info.add_single_fp_config(proto::DeviceFPConfig::SoftFloat);
  }
  fpc = CLInfo<CL_DEVICE_DOUBLE_FP_CONFIG>(did);
  if (fpc & CL_FP_DENORM) {
    info.add_double_fp_config(proto::DeviceFPConfig::Denorm);
  }
  if (fpc & CL_FP_INF_NAN) {
    info.add_double_fp_config(proto::DeviceFPConfig::InfNan);
  }
  if (fpc & CL_FP_ROUND_TO_NEAREST) {
    info.add_double_fp_config(proto::DeviceFPConfig::RoundToNearest);
  }
  if (fpc & CL_FP_ROUND_TO_ZERO) {
    info.add_double_fp_config(proto::DeviceFPConfig::RoundToZero);
  }
  if (fpc & CL_FP_ROUND_TO_INF) {
    info.add_double_fp_config(proto::DeviceFPConfig::RoundToInf);
  }
  if (fpc & CL_FP_FMA) {
    info.add_double_fp_config(proto::DeviceFPConfig::FusedMultiplyAdd);
  }
  if (fpc & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) {
    info.add_double_fp_config(proto::DeviceFPConfig::CorrectlyRoundedDivideSqrt);
  }
  if (fpc & CL_FP_SOFT_FLOAT) {
    info.add_double_fp_config(proto::DeviceFPConfig::SoftFloat);
  }
  switch (CLInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>(did)) {
    case CL_NONE:
      info.set_global_mem_cache_type(proto::MemCacheType::None);
      break;
    case CL_READ_ONLY_CACHE:
      info.set_global_mem_cache_type(proto::MemCacheType::ReadOnly);
      break;
    case CL_READ_WRITE_CACHE:
      info.set_global_mem_cache_type(proto::MemCacheType::ReadWrite);
      break;
  }
  info.set_global_mem_cacheline_size(CLInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>(did));
  info.set_global_mem_cache_size(CLInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>(did));
  info.set_global_mem_size(CLInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(did));
  info.set_max_constant_buffer_size(CLInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>(did));
  info.set_max_constant_args(CLInfo<CL_DEVICE_MAX_CONSTANT_ARGS>(did));
#ifndef __APPLE__
  info.set_max_global_variable_size(CLInfo<CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE>(did));
  info.set_global_variable_preferred_total_size(CLInfo<CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE>(did));
#endif
  switch (CLInfo<CL_DEVICE_LOCAL_MEM_TYPE>(did)) {
    case CL_NONE:
      info.set_local_mem_type(proto::LocalMemType::None);
      break;
    case CL_LOCAL:
      info.set_local_mem_type(proto::LocalMemType::Local);
      break;
    case CL_GLOBAL:
      info.set_local_mem_type(proto::LocalMemType::Global);
      break;
  }
  info.set_local_mem_size(CLInfo<CL_DEVICE_LOCAL_MEM_SIZE>(did));
  info.set_host_unified_memory(CLInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>(did));
  info.set_error_correction_support(CLInfo<CL_DEVICE_ERROR_CORRECTION_SUPPORT>(did));
  info.set_profiling_timer_resolution_ns(CLInfo<CL_DEVICE_PROFILING_TIMER_RESOLUTION>(did));
  info.set_endian_little(CLInfo<CL_DEVICE_ENDIAN_LITTLE>(did));
  info.set_available(CLInfo<CL_DEVICE_AVAILABLE>(did));
  info.set_compiler_available(CLInfo<CL_DEVICE_COMPILER_AVAILABLE>(did));
  info.set_linker_available(CLInfo<CL_DEVICE_LINKER_AVAILABLE>(did));
  auto exec_cap = CLInfo<CL_DEVICE_EXECUTION_CAPABILITIES>(did);
  if (exec_cap & CL_EXEC_KERNEL) {
    info.add_execution_capability(proto::DeviceExecutionCapability::Kernel);
  }
  if (exec_cap & CL_EXEC_NATIVE_KERNEL) {
    info.add_execution_capability(proto::DeviceExecutionCapability::NativeKernel);
  }
#ifndef __APPLE__
  auto cqprop = CLInfo<CL_DEVICE_QUEUE_ON_HOST_PROPERTIES>(did);
  if (cqprop & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
    info.add_queue_on_host_property(proto::CommandQueueProperty::OutOfOrderExecModeEnable);
  }
  if (cqprop & CL_QUEUE_PROFILING_ENABLE) {
    info.add_queue_on_host_property(proto::CommandQueueProperty::ProfilingEnable);
  }
  cqprop = CLInfo<CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES>(did);
  if (cqprop & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
    info.add_queue_on_device_property(proto::CommandQueueProperty::OutOfOrderExecModeEnable);
  }
  if (cqprop & CL_QUEUE_PROFILING_ENABLE) {
    info.add_queue_on_device_property(proto::CommandQueueProperty::ProfilingEnable);
  }
  info.set_queue_on_device_preferred_size(CLInfo<CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE>(did));
  info.set_queue_on_device_max_size(CLInfo<CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE>(did));
  info.set_max_on_device_queues(CLInfo<CL_DEVICE_MAX_ON_DEVICE_QUEUES>(did));
  info.set_max_on_device_events(CLInfo<CL_DEVICE_MAX_ON_DEVICE_EVENTS>(did));
  ForEachElt(CLInfo<CL_DEVICE_BUILT_IN_KERNELS>(did),
             [&info](std::string kernel) { info.add_built_in_kernel(std::move(kernel)); }, boost::regex{R"([^;]+)"});
#endif
  info.set_platform_index(pidx);
  {
    // Strip leading and trailing whitespace, because some vendors want to include it in their device names.
    std::string name = CLInfo<CL_DEVICE_NAME>(did);
    if (info.type() == hal::proto::HardwareType::CPU) {
      name = "CPU";
    }
    auto name_begin = name.begin();
    auto name_end = name.end();
    while (name_begin != name_end && std::isspace(*name_begin)) {
      ++name_begin;
    }
    while (name_begin != name_end && std::isspace(name_end[-1])) {
      --name_end;
    }
    info.set_name(std::string(name_begin, name_end));
  }
  info.set_vendor(CLInfo<CL_DEVICE_VENDOR>(did));
  info.set_driver_version(CLInfo<CL_DRIVER_VERSION>(did));
  info.set_profile(CLInfo<CL_DEVICE_PROFILE>(did));
  info.set_version(CLInfo<CL_DEVICE_VERSION>(did));
  info.set_opencl_c_version(CLInfo<CL_DEVICE_OPENCL_C_VERSION>(did));
  std::set<std::string> exts;
  ForEachElt(CLInfo<CL_DEVICE_EXTENSIONS>(did), [&info, &exts](std::string ext) {
    info.add_extension(ext);
    exts.emplace(std::move(ext));
  });
  info.set_printf_buffer_size(CLInfo<CL_DEVICE_PRINTF_BUFFER_SIZE>(did));
  info.set_preferred_interop_user_sync(CLInfo<CL_DEVICE_PREFERRED_INTEROP_USER_SYNC>(did));
  info.set_partition_max_sub_devices(CLInfo<CL_DEVICE_PARTITION_MAX_SUB_DEVICES>(did));
  for (auto prop : CLInfo<CL_DEVICE_PARTITION_PROPERTIES>(did)) {
    switch (prop) {
      case CL_DEVICE_PARTITION_EQUALLY:
        info.add_device_partition_property(proto::DevicePartitionProperty::Equally);
        break;
      case CL_DEVICE_PARTITION_BY_COUNTS:
        info.add_device_partition_property(proto::DevicePartitionProperty::ByCounts);
        break;
      case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
        info.add_device_partition_property(proto::DevicePartitionProperty::ByAffinityDomain);
        break;
    }
  }
  info.set_partition_affinity_domain(CLInfo<CL_DEVICE_PARTITION_AFFINITY_DOMAIN>(did));
  for (auto ptype : CLInfo<CL_DEVICE_PARTITION_TYPE>(did)) {
    switch (ptype) {
      case CL_DEVICE_AFFINITY_DOMAIN_NUMA:
        info.add_device_partition_type(proto::DevicePartitionType::NUMA);
        break;
      case CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE:
        info.add_device_partition_type(proto::DevicePartitionType::L4Cache);
        break;
      case CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE:
        info.add_device_partition_type(proto::DevicePartitionType::L3Cache);
        break;
      case CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE:
        info.add_device_partition_type(proto::DevicePartitionType::L2Cache);
        break;
      case CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE:
        info.add_device_partition_type(proto::DevicePartitionType::L1Cache);
        break;
      case CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE:
        info.add_device_partition_type(proto::DevicePartitionType::NextPartitionable);
        break;
    }
  }
#ifndef __APPLE__
  auto svm_cap = CLInfo<CL_DEVICE_SVM_CAPABILITIES>(did);
  if (svm_cap & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
    info.add_svm_capability(proto::SvmCapability::CoarseGrainBuffer);
  }
  if (svm_cap & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) {
    info.add_svm_capability(proto::SvmCapability::FineGrainBuffer);
  }
  if (svm_cap & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) {
    info.add_svm_capability(proto::SvmCapability::FineGrainSystem);
  }
  if (svm_cap & CL_DEVICE_SVM_ATOMICS) {
    info.add_svm_capability(proto::SvmCapability::Atomics);
  }
  info.set_preferred_platform_atomic_alignment(CLInfo<CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT>(did));
  info.set_preferred_global_atomic_alignment(CLInfo<CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT>(did));
  info.set_preferred_local_atomic_alignment(CLInfo<CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT>(did));
  info.set_max_num_sub_groups(CLInfo<CL_DEVICE_MAX_NUM_SUB_GROUPS>(did));
  info.set_sub_group_independent_forward_progress(CLInfo<CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS>(did));
#endif

  info.set_platform_name(pinfo.name());

#ifdef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
  if (exts.find("cl_nv_device_attribute_query") != exts.end()) {
    info.set_nv_cuda_major_rev(CLInfo<CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV>(did));
    info.set_nv_cuda_minor_rev(CLInfo<CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV>(did));
    info.set_nv_registers_per_block(CLInfo<CL_DEVICE_REGISTERS_PER_BLOCK_NV>(did));
    info.set_nv_warp_size(CLInfo<CL_DEVICE_WARP_SIZE_NV>(did));
    info.set_nv_gpu_overlap(CLInfo<CL_DEVICE_GPU_OVERLAP_NV>(did));
    info.set_nv_kernel_exec_timeout(CLInfo<CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV>(did));
    info.set_nv_integrated_memory(CLInfo<CL_DEVICE_INTEGRATED_MEMORY_NV>(did));
  }
#endif
#ifdef CL_DEVICE_BOARD_NAME_AMD
  if (exts.find("cl_amd_device_attribute_query") != exts.end()) {
    info.set_amd_board_name(CLInfo<CL_DEVICE_BOARD_NAME_AMD>(did));
    info.set_amd_simd_per_compute_unit(CLInfo<CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD>(did));
    info.set_amd_simd_instruction_width(CLInfo<CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD>(did));
    info.set_amd_wavefront_width(CLInfo<CL_DEVICE_WAVEFRONT_WIDTH_AMD>(did));
    info.set_amd_global_mem_channels(CLInfo<CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD>(did));
    info.set_amd_global_mem_channel_banks(CLInfo<CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD>(did));
    info.set_amd_global_mem_channel_bank_width(CLInfo<CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD>(did));
    info.set_amd_local_mem_size_per_compute_unit(CLInfo<CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD>(did));
    info.set_amd_local_mem_banks(CLInfo<CL_DEVICE_LOCAL_MEM_BANKS_AMD>(did));
  }
#endif

  return info;
}

}  // namespace

DeviceSet::DeviceSet(const context::Context& ctx, std::uint32_t pidx, cl_platform_id pid) {
  context::Activity platform_activity{ctx, "tile::hal::opencl::Platform"};
  auto pinfo = GetPlatformInfo(pid);
  LogInfo(std::string("Platform[") + std::to_string(pidx) + "]", pinfo);
  platform_activity.AddMetadata(pinfo);

  cl_uint device_count;
  clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
  std::vector<cl_device_id> devices(device_count);
  clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, devices.size(), devices.data(), nullptr);

  cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(pid), 0};
  Err err;
  CLObj<cl_context> cl_ctx = clCreateContext(props, devices.size(), devices.data(), &OnErr, nullptr, err.ptr());
  if (!cl_ctx) {
    throw std::runtime_error(std::string("failed to create a context for OpenCL devices on platform ") +
                             CLInfo<CL_PLATFORM_NAME>(pid) + ": " + err.str());
  }

  std::shared_ptr<Device> first_dev;

  for (std::uint32_t didx = 0; didx < devices.size(); ++didx) {
    context::Activity device_activity{platform_activity.ctx(), "tile::hal::opencl::Device"};
    auto did = devices[didx];
    auto dinfo = GetDeviceInfo(did, pidx, pinfo);
    *dinfo.mutable_platform_id() = platform_activity.ctx().activity_id();
    LogInfo(std::string("Platform[") + std::to_string(pidx) + "].Device[" + std::to_string(didx) + "]", dinfo);
    device_activity.AddMetadata(dinfo);
    auto dev = std::make_shared<Device>(device_activity.ctx(), cl_ctx, did, std::move(dinfo));
    if (!first_dev) {
      first_dev = dev;
    }
    devices_.emplace_back(std::move(dev));
  }

  if (first_dev) {
    host_memory_ = compat::make_unique<HostMemory>(first_dev->device_state());
  }
}

const std::vector<std::shared_ptr<hal::Device>>& DeviceSet::devices() { return devices_; }

Memory* DeviceSet::host_memory() { return host_memory_.get(); }

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
