/*
 *
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "test_harness/test_harness.hpp"

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

void zeEventPool::InitEventPool() { InitEventPool(32); }

void zeEventPool::InitEventPool(uint32_t count) {
  InitEventPool(count, ZE_EVENT_POOL_FLAG_HOST_VISIBLE);
}
void zeEventPool::InitEventPool(uint32_t count, ze_event_pool_flags_t flags) {
  if (event_pool_ == nullptr) {
    if (context_ == nullptr) {
      context_ = lzt::get_default_context();
    }
    event_pool_ = create_event_pool(context_, count, flags);
    pool_indexes_available_.resize(count, true);
  }
}

void zeEventPool::InitEventPool(ze_context_handle_t context, uint32_t count) {
  context_ = context;
  if (event_pool_ == nullptr) {
    if (context_ == nullptr) {
      context_ = lzt::get_default_context();
    }
    event_pool_ =
        create_event_pool(context_, count, ZE_EVENT_POOL_FLAG_HOST_VISIBLE);
    pool_indexes_available_.resize(count, true);
  }
}

void zeEventPool::InitEventPool(ze_event_pool_desc_t desc) {
  if (event_pool_ == nullptr) {
    if (context_ == nullptr) {
      context_ = lzt::get_default_context();
    }
    event_pool_ = create_event_pool(context_, desc);
    pool_indexes_available_.resize(desc.count, true);
  }
}

void zeEventPool::InitEventPool(ze_event_pool_desc_t desc,
                                std::vector<ze_device_handle_t> devices) {
  if (event_pool_ == nullptr) {
    if (context_ == nullptr) {
      context_ = lzt::get_default_context();
    }
    event_pool_ = create_event_pool(context_, desc, devices);
    pool_indexes_available_.resize(desc.count, true);
  }
}

zeEventPool::zeEventPool() {}

zeEventPool::~zeEventPool() {
  // If the event pool was never created, do not attempt to destroy it
  // as that will needlessly cause a test failure.
  if (event_pool_) {
    destroy_event_pool(event_pool_);
  }
}

uint32_t find_index(const std::vector<bool> &indexes_available) {
  for (uint32_t i = 0; i < indexes_available.size(); i++)
    if (indexes_available[i])
      return i;
  return -1;
}

void zeEventPool::create_event(ze_event_handle_t &event) {
  create_event(event, 0, 0);
}

void zeEventPool::create_event(ze_event_handle_t &event,
                               ze_event_scope_flags_t signal,
                               ze_event_scope_flags_t wait) {
  // Make sure the event pool is initialized to at least defaults:
  InitEventPool();
  ze_event_desc_t desc = {};
  memset(&desc, 0, sizeof(desc));
  desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
  desc.pNext = nullptr;
  desc.signal = signal;
  desc.wait = wait;
  event = nullptr;
  desc.index = find_index(pool_indexes_available_);
  EXPECT_EQ(ZE_RESULT_SUCCESS, zeEventCreate(event_pool_, &desc, &event));
  EXPECT_NE(nullptr, event);
  handle_to_index_map_[event] = desc.index;
  pool_indexes_available_[desc.index] = false;
}

// Use to bypass zeEventPool management of event indexes
void zeEventPool::create_event(ze_event_handle_t &event, ze_event_desc_t desc) {
  EXPECT_EQ(ZE_RESULT_SUCCESS, zeEventCreate(event_pool_, &desc, &event));
  handle_to_index_map_[event] = desc.index;
  pool_indexes_available_[desc.index] = false;
}

void zeEventPool::create_events(std::vector<ze_event_handle_t> &events,
                                size_t event_count) {
  create_events(events, event_count, 0, 0);
}

void zeEventPool::create_events(std::vector<ze_event_handle_t> &events,
                                size_t event_count,
                                ze_event_scope_flags_t signal,
                                ze_event_scope_flags_t wait) {
  events.resize(event_count);
  for (auto &event : events)
    create_event(event, signal, wait);
}

void zeEventPool::destroy_event(ze_event_handle_t event) {
  std::map<ze_event_handle_t, uint32_t>::iterator it =
      handle_to_index_map_.find(event);

  EXPECT_NE(it, handle_to_index_map_.end());
  pool_indexes_available_[(*it).second] = true;
  handle_to_index_map_.erase(it);
  EXPECT_EQ(ZE_RESULT_SUCCESS, zeEventDestroy(event));
}

void zeEventPool::destroy_events(std::vector<ze_event_handle_t> &events) {
  for (auto &event : events)
    destroy_event(event);
  events.clear();
}

void zeEventPool::get_ipc_handle(ze_ipc_event_pool_handle_t *hIpc) {
  ASSERT_EQ(ZE_RESULT_SUCCESS, zeEventPoolGetIpcHandle(event_pool_, hIpc));
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
