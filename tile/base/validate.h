// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <cstdint>

namespace vertexai {
namespace tile {

// Validates memory access bounds, throwing an exception on negative values and
// integer overflow.
void ValidateBounds(std::int64_t count, std::int64_t offset);

// Validates memory access bounds, throwing an exception on negative values and
// integer overflow,
// as well as validating that the access is contained within the requested size.
void ValidateBounds(std::int64_t count, std::int64_t offset, std::int64_t size);

}  // namespace tile
}  // namespace vertexai
