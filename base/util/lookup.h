#pragma once

#include <map>

#include "base/util/printstring.h"
#include "base/util/throw.h"

namespace vertexai {

template <typename M>
const typename M::mapped_type& safe_at(const M& map, const typename M::key_type& key) {
  auto it = map.find(key);
  if (it == map.end()) {
    throw_with_trace(std::runtime_error(printstring("Key not found: %s", to_string(key).c_str())));
  }
  return it->second;
}

}  // namespace vertexai
