// Copyright 2017-2018 Intel Corporation.

#include "tile/base/validate.h"

#include <cstdint>

#include "base/util/error.h"

namespace vertexai {
namespace tile {

void ValidateBounds(int64_t count, int64_t offset) {
  if (offset < 0) {
    throw error::InvalidArgument("memory access offsets must be >= 0");
  }

  if (count < 0) {
    throw error::InvalidArgument("memory access lengths must be >= 0");
  }

  if (offset > (INT64_MAX - count)) {
    throw error::OutOfRange("offset+count overflow");
  }
}

void ValidateBounds(int64_t count, int64_t offset, int64_t size) {
  ValidateBounds(count, offset);

  if (size < offset + count) {
    throw error::OutOfRange("unable to access memory past the end of a buffer");
  }
}

}  // namespace tile
}  // namespace vertexai
