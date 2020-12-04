/*
 *
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef level_zero_tests_ZE_TEST_HARNESS_DEVICE_HPP
#define level_zero_tests_ZE_TEST_HARNESS_DEVICE_HPP

#include <level_zero/ze_api.h>
#include <mutex>
#include <vector>

namespace level_zero_tests {

class zeDevice {
public:
  static zeDevice *get_instance();
  ze_device_handle_t get_device();
  ze_driver_handle_t get_driver();

private:
  zeDevice();
  static zeDevice *instance_;
  static std::once_flag instance;
  ze_device_handle_t device_ = nullptr;
  ze_driver_handle_t driver_ = nullptr;
};

uint32_t get_ze_device_count();
uint32_t get_ze_device_count(ze_driver_handle_t driver);
std::vector<ze_device_handle_t> get_ze_devices();
std::vector<ze_device_handle_t> get_ze_devices(uint32_t count);
std::vector<ze_device_handle_t> get_ze_devices(ze_driver_handle_t driver);
std::vector<ze_device_handle_t> get_ze_devices(uint32_t count,
                                               ze_driver_handle_t driver);
uint32_t get_ze_sub_device_count(ze_device_handle_t device);
std::vector<ze_device_handle_t> get_ze_sub_devices(ze_device_handle_t device);
std::vector<ze_device_handle_t> get_ze_sub_devices(ze_device_handle_t device,
                                                   uint32_t count);

ze_device_properties_t get_device_properties(ze_device_handle_t device);
uint32_t get_command_queue_group_properties_count(ze_device_handle_t device);
std::vector<ze_command_queue_group_properties_t>
get_command_queue_group_properties(ze_device_handle_t device);
std::vector<ze_command_queue_group_properties_t>
get_command_queue_group_properties(ze_device_handle_t device, uint32_t count);
ze_device_compute_properties_t
get_compute_properties(ze_device_handle_t device);
uint32_t get_memory_properties_count(ze_device_handle_t device);
std::vector<ze_device_memory_properties_t>
get_memory_properties(ze_device_handle_t device);
std::vector<ze_device_memory_properties_t>
get_memory_properties(ze_device_handle_t device, uint32_t count);
ze_device_external_memory_properties_t
get_external_memory_properties(ze_device_handle_t device);
ze_device_memory_access_properties_t
get_memory_access_properties(ze_device_handle_t device);
std::vector<ze_device_cache_properties_t>
get_cache_properties(ze_device_handle_t device);
ze_device_image_properties_t get_image_properties(ze_device_handle_t device);
ze_device_module_properties_t
get_device_module_properties(ze_device_handle_t device);

ze_device_p2p_properties_t get_p2p_properties(ze_device_handle_t dev1,
                                              ze_device_handle_t dev2);
ze_bool_t can_access_peer(ze_device_handle_t dev1, ze_device_handle_t dev2);

void set_kernel_cache_config(ze_kernel_handle_t kernel,
                             ze_cache_config_flags_t config);

void make_memory_resident(const ze_device_handle_t &device, void *memory,
                          const size_t size);
void evict_memory(const ze_device_handle_t &device, void *memory,
                  const size_t size);

}; // namespace level_zero_tests
#endif
