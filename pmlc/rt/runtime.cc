// Copyright 2020, Intel Corporation

#include "pmlc/rt/runtime.h"

#include "pmlc/rt/internal.h"

namespace pmlc::rt {
namespace {

thread_local std::shared_ptr<Device> currentDevice;

} // namespace

ScopedCurrentDevice::ScopedCurrentDevice(std::shared_ptr<Device> device) {
  currentDevice = device;
}

ScopedCurrentDevice::~ScopedCurrentDevice() { currentDevice = nullptr; }

std::shared_ptr<Device> Device::currentUntyped() { return currentDevice; }

} // namespace pmlc::rt
