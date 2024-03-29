// Copyright 2018 Intel Corporation

#include "pmlc/util/env.h"

#if _MSC_VER
#include <windows.h>
#endif

#include <cstdlib>

namespace pmlc::util {

std::string getEnvVar(const std::string &key,
                      const std::string &default_value) {
#ifdef _MSC_VER
  char var[1024];
  auto rv = GetEnvironmentVariableA(key.c_str(), var, sizeof(var));
  if (!rv || sizeof(var) <= rv) {
    return default_value;
  }
  return var;
#else
  char const *val = std::getenv(key.c_str());
  return val == nullptr ? default_value : val;
#endif
}

void setEnvVar(const std::string &key, const std::string &value) {
#ifdef _MSC_VER
  SetEnvironmentVariableA(key.c_str(), value.c_str());
#else
  setenv(key.c_str(), value.c_str(), 1);
#endif
}

} // namespace pmlc::util
