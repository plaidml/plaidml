// Copyright Vertex.AI.

#include "base/util/env.h"

#ifdef _MSC_VER
#include <windows.h>
#endif

#include <cstdlib>

namespace vertexai {
namespace env {

std::string GetVar(std::string const& key) {
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

}  // namespace env
}  // namespace vertexai
