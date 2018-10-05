// Copyright 2018 Intel Corporation.

#include <exception>

#include "base/util/perf_counter.h"
#include "plaidml/base/base.h"
#include "plaidml/base/status.h"

extern "C" VAI_API int64_t vai_get_perf_counter(const char* name) {
  try {
    return vertexai::GetPerfCounter(name);
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return -1;
  }
}

extern "C" VAI_API void vai_set_perf_counter(const char* name, int64_t value) {
  try {
    vertexai::SetPerfCounter(name, value);
  } catch (...) {
    // Ignore error
  }
}
