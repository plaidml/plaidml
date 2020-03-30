// Copyright 2020 Intel Corporation

#include <stdint.h>

#include <stdexcept>

extern "C" void plaidml_rt_bounds_check(intptr_t index, unsigned range) {
  int64_t accessIndex = (int64_t)index;
  int64_t upperRange = (int64_t)range;
  if (accessIndex < 0 || accessIndex >= upperRange)
    std::runtime_error("Out of bounds index for mlir::LoadOp or mlir::StoreOp");
}
