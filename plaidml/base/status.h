// Copyright 2018 Intel Corporation.
//
// Internal Vertex.AI base library definitions.

#pragma once

#include <exception>

#include "plaidml/base/base.h"

namespace vertexai {

// Sets the thread's thread-local status.  The supplied string will be copied by the
// library before returning to the caller.  The string should be UTF-8, and may depend
// on the locale installed when the error occurred.
void SetLastStatus(vai_status status, const char* str) noexcept;

// Sets the thread's thread-local status based on the supplied exception.
void SetLastException(std::exception_ptr ep) noexcept;

// Sets the thread's thread-local status to indicate an out-of-memory condition.
void SetLastOOM() noexcept;

}  // namespace vertexai
