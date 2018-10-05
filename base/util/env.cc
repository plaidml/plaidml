// Copyright 2018 Intel Corporation

#include "base/util/env.h"

#if _MSC_VER
#include <windows.h>
#endif

#include <cstdlib>

namespace vertexai {
namespace env {

std::string Get(std::string key) {
#ifdef _MSC_VER
  char var[1024];
  auto rv = GetEnvironmentVariableA(key.c_str(), var, sizeof(var));
  if (!rv || sizeof(var) <= rv) {
    return "";
  }
  return std::string(var);
#else
  char const* val = std::getenv(key.c_str());
  return val == nullptr ? "" : val;
#endif
}

void Set(std::string key, std::string value) {
#ifdef _MSC_VER
  SetEnvironmentVariableA(key.c_str(), value.c_str());
#else
  setenv(key.c_str(), value.c_str(), 1);
#endif
}

}  // namespace env
}  // namespace vertexai
