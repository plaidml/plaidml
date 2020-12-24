// Copyright 2020 Intel Corporation
#ifndef PMLC_RT_LEVEL_ZERO_LEVEL_ZERO_UTILS_H_
#define PMLC_RT_LEVEL_ZERO_LEVEL_ZERO_UTILS_H_

#include <string.h>
#ifdef PML_USE_DEVICE_LEVEL_ZERO
#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>
#else
#include <ze_api.h>
#include <zet_api.h>
#endif

#include <array>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pmlc::rt::level_zero::lzu {

std::vector<std::pair<ze_driver_handle_t, ze_device_handle_t>>
getSupportedDevices();

// Context
ze_context_handle_t get_context(ze_driver_handle_t driver);

void destroy_context(ze_context_handle_t context);

// Driver
uint32_t get_driver_handle_count();
// Context
ze_context_handle_t get_context(ze_driver_handle_t driver);

void destroy_context(ze_context_handle_t context);

// Driver
uint32_t get_driver_handle_count();

std::vector<ze_driver_handle_t> get_all_driver_handles();

// Device
uint32_t get_device_count(ze_driver_handle_t driver);

std::vector<ze_device_handle_t> get_devices(ze_driver_handle_t driver);

ze_device_properties_t get_device_properties(ze_device_handle_t device);

// Memory
void *allocate_host_memory(const size_t size, const size_t alignment,
                           const ze_context_handle_t context);

void *allocate_device_memory(const size_t size, const size_t alignment,
                             const ze_device_mem_alloc_flags_t flags,
                             const uint32_t ordinal,
                             ze_device_handle_t device_handle,
                             ze_context_handle_t context);

void *allocate_shared_memory(const size_t size, const size_t alignment,
                             const ze_device_mem_alloc_flags_t dev_flags,
                             const ze_host_mem_alloc_flags_t host_flags,
                             ze_device_handle_t device,
                             ze_context_handle_t context);

void free_memory(ze_context_handle_t context, void *ptr);

void append_memory_copy(ze_command_list_handle_t cl, void *dstptr,
                        const void *srcptr, size_t size,
                        ze_event_handle_t hSignalEvent,
                        uint32_t num_wait_events,
                        ze_event_handle_t *wait_events);

// Module
ze_module_handle_t create_module(ze_context_handle_t context,
                                 ze_device_handle_t device, uint8_t *data,
                                 size_t bytes, const ze_module_format_t format,
                                 const char *build_flags,
                                 ze_module_build_log_handle_t *p_build_log);

void destroy_module(ze_module_handle_t module);

// Kernel
ze_kernel_handle_t create_function(ze_module_handle_t module,
                                   ze_kernel_flags_t flag,
                                   std::string func_name);

void set_argument_value(ze_kernel_handle_t hFunction, uint32_t argIndex,
                        size_t argSize, const void *pArgValue);

void append_launch_function(ze_command_list_handle_t hCommandList,
                            ze_kernel_handle_t hFunction,
                            const ze_group_count_t *pLaunchFuncArgs,
                            ze_event_handle_t hSignalEvent,
                            uint32_t numWaitEvents,
                            ze_event_handle_t *phWaitEvents);

void destroy_function(ze_kernel_handle_t kernel);

// Command list
ze_command_list_handle_t create_command_list(ze_context_handle_t context,
                                             ze_device_handle_t device,
                                             ze_command_list_flags_t flags,
                                             uint32_t ordinal);

void close_command_list(ze_command_list_handle_t cl);

void execute_command_lists(ze_command_queue_handle_t cq,
                           uint32_t numCommandLists,
                           ze_command_list_handle_t *phCommandLists,
                           ze_fence_handle_t hFence);

void reset_command_list(ze_command_list_handle_t cl);

void destroy_command_list(ze_command_list_handle_t cl);

// Command queue
ze_command_queue_handle_t create_command_queue(
    ze_context_handle_t context, ze_device_handle_t device,
    ze_command_queue_flags_t flags, ze_command_queue_mode_t mode,
    ze_command_queue_priority_t priority, uint32_t ordinal, uint32_t index);

void synchronize(ze_command_queue_handle_t cq, uint64_t timeout);

void destroy_command_queue(ze_command_queue_handle_t cq);

// Event
class zeEventPool {
public:
  zeEventPool();
  ~zeEventPool();

  void
  InitEventPool(ze_context_handle_t context, uint32_t count,
                ze_event_pool_flags_t flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE);

  void create_event(ze_event_handle_t &event, ze_event_scope_flags_t signal = 0,
                    ze_event_scope_flags_t wait = 0);

  void destroy_event(ze_event_handle_t event);

  ze_event_pool_handle_t event_pool_ = nullptr;
  ze_context_handle_t context_ = nullptr;
  std::vector<bool> pool_indexes_available_;
  std::map<ze_event_handle_t, uint32_t> handle_to_index_map_;
};

void append_barrier(ze_command_list_handle_t cl, ze_event_handle_t hSignalEvent,
                    uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);

// Group
void set_group_size(ze_kernel_handle_t hFunction, uint32_t groupSizeX,
                    uint32_t groupSizeY, uint32_t groupSizeZ);

void suggest_group_size(ze_kernel_handle_t hFunction, uint32_t globalSizeX,
                        uint32_t globalSizeY, uint32_t globalSizeZ,
                        uint32_t &groupSizeX, uint32_t &groupSizeY,
                        uint32_t &groupSizeZ);

// Helper
std::string to_string(const ze_result_t result);

} // namespace pmlc::rt::level_zero::lzu

#endif // PMLC_RT_LEVEL_ZERO_LEVEL_ZERO_UTILS_H_
