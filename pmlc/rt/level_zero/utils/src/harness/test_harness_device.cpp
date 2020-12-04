/*
 *
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "test_harness/test_harness_device.hpp"
#include "utils/utils.hpp"
#include <level_zero/ze_api.h>

namespace lzt = level_zero_tests;

namespace level_zero_tests {

zeDevice *zeDevice::instance_ = nullptr;
std::once_flag zeDevice::instance;

zeDevice *zeDevice::get_instance() {
  std::call_once(instance, []() {
    instance_ = new zeDevice;
    EXPECT_EQ(ZE_RESULT_SUCCESS, zeInit(0));

    instance_->driver_ = lzt::get_default_driver();
    instance_->device_ = lzt::get_default_device(instance_->driver_);
  });
  return instance_;
}

ze_device_handle_t zeDevice::get_device() { return get_instance()->device_; }
ze_driver_handle_t zeDevice::get_driver() { return get_instance()->driver_; }

zeDevice::zeDevice() {
  device_ = nullptr;
  driver_ = nullptr;
}

uint32_t get_ze_device_count() {
  return get_ze_device_count(lzt::get_default_driver());
}

uint32_t get_ze_device_count(ze_driver_handle_t driver) {
  uint32_t count = 0;
  EXPECT_EQ(ZE_RESULT_SUCCESS, zeDeviceGet(driver, &count, nullptr));

  return count;
}

std::vector<ze_device_handle_t> get_ze_devices() {
  return get_ze_devices(get_ze_device_count());
}

std::vector<ze_device_handle_t> get_ze_devices(uint32_t count) {
  return get_ze_devices(count, lzt::get_default_driver());
}

std::vector<ze_device_handle_t> get_ze_devices(ze_driver_handle_t driver) {
  return get_ze_devices(get_ze_device_count(driver), driver);
}

std::vector<ze_device_handle_t> get_ze_devices(uint32_t count,
                                               ze_driver_handle_t driver) {
  uint32_t count_out = count;
  std::vector<ze_device_handle_t> devices(count);

  EXPECT_EQ(ZE_RESULT_SUCCESS, zeDeviceGet(driver, &count_out, devices.data()));
  if (count == get_ze_device_count())
    EXPECT_EQ(count_out, count);

  return devices;
}

uint32_t get_ze_sub_device_count(ze_device_handle_t device) {
  uint32_t count = 0;

  EXPECT_EQ(ZE_RESULT_SUCCESS, zeDeviceGetSubDevices(device, &count, nullptr));
  return count;
}

std::vector<ze_device_handle_t> get_ze_sub_devices(ze_device_handle_t device) {
  return get_ze_sub_devices(device, get_ze_sub_device_count(device));
}

std::vector<ze_device_handle_t> get_ze_sub_devices(ze_device_handle_t device,
                                                   uint32_t count) {
  std::vector<ze_device_handle_t> sub_devices(count);

  EXPECT_EQ(ZE_RESULT_SUCCESS,
            zeDeviceGetSubDevices(device, &count, sub_devices.data()));
  return sub_devices;
}

ze_device_properties_t get_device_properties(ze_device_handle_t device) {
  ze_device_properties_t properties = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};

  EXPECT_EQ(ZE_RESULT_SUCCESS, zeDeviceGetProperties(device, &properties));
  return properties;
}

ze_device_compute_properties_t
get_compute_properties(ze_device_handle_t device) {
  ze_device_compute_properties_t properties = {
      ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES};

  EXPECT_EQ(ZE_RESULT_SUCCESS,
            zeDeviceGetComputeProperties(device, &properties));
  return properties;
}

uint32_t get_memory_properties_count(ze_device_handle_t device) {
  uint32_t count = 0;

  EXPECT_EQ(ZE_RESULT_SUCCESS,
            zeDeviceGetMemoryProperties(device, &count, nullptr));
  return count;
}

std::vector<ze_device_memory_properties_t>
get_memory_properties(ze_device_handle_t device) {
  return get_memory_properties(device, get_memory_properties_count(device));
}

std::vector<ze_device_memory_properties_t>
get_memory_properties(ze_device_handle_t device, uint32_t count) {
  std::vector<ze_device_memory_properties_t> properties(
      count, {ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES});

  EXPECT_EQ(ZE_RESULT_SUCCESS,
            zeDeviceGetMemoryProperties(device, &count, properties.data()));
  return properties;
}

ze_device_external_memory_properties_t
get_external_memory_properties(ze_device_handle_t device) {
  ze_device_external_memory_properties_t properties = {
      ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES};

  EXPECT_EQ(ZE_RESULT_SUCCESS,
            zeDeviceGetExternalMemoryProperties(device, &properties));

  return properties;
}

ze_device_memory_access_properties_t
get_memory_access_properties(ze_device_handle_t device) {
  ze_device_memory_access_properties_t properties = {
      ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES};

  EXPECT_EQ(ZE_RESULT_SUCCESS,
            zeDeviceGetMemoryAccessProperties(device, &properties));
  return properties;
}

uint32_t get_command_queue_group_properties_count(ze_device_handle_t device) {
  uint32_t count = 0;

  EXPECT_EQ(ZE_RESULT_SUCCESS,
            zeDeviceGetCommandQueueGroupProperties(device, &count, nullptr));

  return count;
}

std::vector<ze_command_queue_group_properties_t>
get_command_queue_group_properties(ze_device_handle_t device, uint32_t count) {
  std::vector<ze_command_queue_group_properties_t> properties(count);

  for (auto properties_item : properties) {
    properties_item.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  }
  EXPECT_EQ(ZE_RESULT_SUCCESS, zeDeviceGetCommandQueueGroupProperties(
                                   device, &count, properties.data()));
  return properties;
}

std::vector<ze_command_queue_group_properties_t>
get_command_queue_group_properties(ze_device_handle_t device) {
  return get_command_queue_group_properties(
      device, get_command_queue_group_properties_count(device));
}

std::vector<ze_device_cache_properties_t>
get_cache_properties(ze_device_handle_t device) {

  std::vector<ze_device_cache_properties_t> properties;
  uint32_t count = 0;
  EXPECT_EQ(ZE_RESULT_SUCCESS,
            zeDeviceGetCacheProperties(device, &count, nullptr));
  properties.resize(count);
  EXPECT_EQ(ZE_RESULT_SUCCESS,
            zeDeviceGetCacheProperties(device, &count, properties.data()));

  return properties;
}

ze_device_image_properties_t get_image_properties(ze_device_handle_t device) {
  ze_device_image_properties_t properties = {
      ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES};

  EXPECT_EQ(ZE_RESULT_SUCCESS, zeDeviceGetImageProperties(device, &properties));
  return properties;
}

ze_device_module_properties_t
get_device_module_properties(ze_device_handle_t device) {
  ze_device_module_properties_t properties = {
      ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES};

  EXPECT_EQ(ZE_RESULT_SUCCESS,
            zeDeviceGetModuleProperties(device, &properties));
  return properties;
}

ze_device_p2p_properties_t get_p2p_properties(ze_device_handle_t dev1,
                                              ze_device_handle_t dev2) {
  ze_device_p2p_properties_t properties = {
      ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES};

  EXPECT_EQ(ZE_RESULT_SUCCESS,
            zeDeviceGetP2PProperties(dev1, dev2, &properties));
  return properties;
}

ze_bool_t can_access_peer(ze_device_handle_t dev1, ze_device_handle_t dev2) {
  ze_bool_t can_access;

  EXPECT_EQ(ZE_RESULT_SUCCESS, zeDeviceCanAccessPeer(dev1, dev2, &can_access));
  return can_access;
}

void set_kernel_cache_config(ze_kernel_handle_t kernel,
                             ze_cache_config_flags_t config) {
  EXPECT_EQ(ZE_RESULT_SUCCESS, zeKernelSetCacheConfig(kernel, config));
}

void make_memory_resident(const ze_device_handle_t &device, void *memory,
                          const size_t size) {
  EXPECT_EQ(
      ZE_RESULT_SUCCESS,
      zeContextMakeMemoryResident(get_default_context(), device, memory, size));
}

void evict_memory(const ze_device_handle_t &device, void *memory,
                  const size_t size) {
  EXPECT_EQ(ZE_RESULT_SUCCESS,
            zeContextEvictMemory(get_default_context(), device, memory, size));
}

}; // namespace level_zero_tests
