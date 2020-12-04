/*
 *
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "test_harness/test_harness_event.hpp"
#include "test_harness/test_harness.hpp"
#include "utils/utils.hpp"

namespace lzt = level_zero_tests;

namespace level_zero_tests {

ze_event_pool_handle_t create_event_pool(ze_context_handle_t context,
                                         uint32_t count,
                                         ze_event_pool_flags_t flags) {
  ze_event_pool_handle_t event_pool;
  ze_event_pool_desc_t descriptor = {};
  descriptor.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;

  descriptor.pNext = nullptr;
  descriptor.flags = flags;
  descriptor.count = count;

  return create_event_pool(context, descriptor);
}

ze_event_pool_handle_t create_event_pool(ze_event_pool_desc_t desc) {

  return create_event_pool(lzt::get_default_context(), desc);
}

ze_event_pool_handle_t create_event_pool(ze_context_handle_t context,
                                         ze_event_pool_desc_t desc) {
  ze_event_pool_handle_t event_pool;
  ze_driver_handle_t driver = lzt::get_default_driver();
  ze_device_handle_t device = zeDevice::get_instance()->get_device();

  EXPECT_EQ(ZE_RESULT_SUCCESS,
            zeEventPoolCreate(context, &desc, 0, nullptr, &event_pool));
  EXPECT_NE(nullptr, event_pool);
  return event_pool;
}

ze_event_pool_handle_t
create_event_pool(ze_context_handle_t context, ze_event_pool_desc_t desc,
                  std::vector<ze_device_handle_t> devices) {
  ze_event_pool_handle_t event_pool;

  EXPECT_EQ(ZE_RESULT_SUCCESS, zeEventPoolCreate(context, &desc, devices.size(),
                                                 devices.data(), &event_pool));
  EXPECT_NE(nullptr, event_pool);
  return event_pool;
}

ze_event_handle_t create_event(ze_event_pool_handle_t event_pool,
                               ze_event_desc_t desc) {
  ze_event_handle_t event;
  EXPECT_EQ(ZE_RESULT_SUCCESS, zeEventCreate(event_pool, &desc, &event));
  EXPECT_NE(nullptr, event);
  return event;
}

void destroy_event(ze_event_handle_t event) {
  EXPECT_EQ(ZE_RESULT_SUCCESS, zeEventDestroy(event));
}

void destroy_event_pool(ze_event_pool_handle_t event_pool) {
  EXPECT_EQ(ZE_RESULT_SUCCESS, zeEventPoolDestroy(event_pool));
}

ze_kernel_timestamp_result_t
get_event_kernel_timestamp(ze_event_handle_t event) {
  // TBD
  ze_kernel_timestamp_result_t value = {};
  EXPECT_EQ(ZE_RESULT_SUCCESS, zeEventQueryKernelTimestamp(event, &value));
  return value;
}

void close_ipc_event_handle(ze_event_pool_handle_t eventPool) {
  EXPECT_EQ(ZE_RESULT_SUCCESS, zeEventPoolCloseIpcHandle(eventPool));
}

void open_ipc_event_handle(ze_context_handle_t context,
                           ze_ipc_event_pool_handle_t hIpc,
                           ze_event_pool_handle_t *eventPool) {
  EXPECT_EQ(ZE_RESULT_SUCCESS,
            zeEventPoolOpenIpcHandle(context, hIpc, eventPool));
}

void signal_event_from_host(ze_event_handle_t hEvent) {
  EXPECT_EQ(ZE_RESULT_SUCCESS, zeEventHostSignal(hEvent));
}

void event_host_synchronize(ze_event_handle_t hEvent, uint64_t timeout) {
  EXPECT_EQ(ZE_RESULT_SUCCESS, zeEventHostSynchronize(hEvent, timeout));
}

void event_host_reset(ze_event_handle_t hEvent) {
  EXPECT_EQ(ZE_RESULT_SUCCESS, zeEventHostReset(hEvent));
}
}; // namespace level_zero_tests
