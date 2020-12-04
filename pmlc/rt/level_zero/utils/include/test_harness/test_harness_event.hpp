/*
 *
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef level_zero_tests_ZE_TEST_HARNESS_EVENT_HPP
#define level_zero_tests_ZE_TEST_HARNESS_EVENT_HPP

#include "test_harness/test_harness.hpp"
#include <level_zero/ze_api.h>

namespace lzt = level_zero_tests;

namespace level_zero_tests {

void signal_event_from_host(ze_event_handle_t hEvent);

ze_event_pool_handle_t create_event_pool(ze_context_handle_t context,
                                         uint32_t count,
                                         ze_event_pool_flags_t flags);
ze_event_pool_handle_t create_event_pool(ze_event_pool_desc_t desc);
ze_event_pool_handle_t create_event_pool(ze_context_handle_t context,
                                         ze_event_pool_desc_t desc);
ze_event_pool_handle_t
create_event_pool(ze_context_handle_t context, ze_event_pool_desc_t desc,
                  std::vector<ze_device_handle_t> devices);
void event_host_synchronize(ze_event_handle_t hEvent, uint64_t timeout);
void event_host_reset(ze_event_handle_t hEvent);
void open_ipc_event_handle(ze_context_handle_t context,
                           ze_ipc_event_pool_handle_t hIpc,
                           ze_event_pool_handle_t *eventPool);
void close_ipc_event_handle(ze_event_pool_handle_t eventPool);

ze_event_handle_t create_event(ze_event_pool_handle_t event_pool,
                               ze_event_desc_t desc);
void destroy_event(ze_event_handle_t event);
void destroy_event_pool(ze_event_pool_handle_t event_pool);
ze_kernel_timestamp_result_t
get_event_kernel_timestamp(ze_event_handle_t event);

}; // namespace level_zero_tests

#endif
