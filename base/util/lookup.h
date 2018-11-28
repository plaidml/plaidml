#pragma once

#include <map>

#include <boost/format.hpp>

#include "base/util/throw.h"

namespace vertexai {

template <typename M>
typename M::mapped_type& safe_at(M* map, const typename M::key_type& key) {
  auto it = map->find(key);
  if (it == map->end()) {
    throw_with_trace(std::runtime_error(str(boost::format("Key not found: %s") % to_string(key))));
  }
  return it->second;
}

template <typename M>
const typename M::mapped_type& safe_at(const M& map, const typename M::key_type& key) {
  auto it = map.find(key);
  if (it == map.end()) {
    throw_with_trace(std::runtime_error(str(boost::format("Key not found: %s") % to_string(key))));
  }
  return it->second;
}

}  // namespace vertexai
