// Copyright 2020, Intel Corporation

#include "pmlc/rt/runtime.h"

#include "pmlc/rt/internal.h"

namespace pmlc::rt {
namespace {

thread_local Device *currentDevice = nullptr;

} // namespace

ScopedCurrentDevice::ScopedCurrentDevice(Device *device) {
  currentDevice = device;
}

ScopedCurrentDevice::~ScopedCurrentDevice() { currentDevice = nullptr; }

Device *Device::currentUntyped() { return currentDevice; }

} // namespace pmlc::rt
