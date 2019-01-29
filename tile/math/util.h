#pragma once

#include <cstdint>

namespace vertexai {
namespace tile {
namespace math {

inline uint64_t NearestPo2(uint64_t x) {
  uint64_t po2 = 1;
  while (po2 < x) {
    po2 *= 2;
  }
  return po2;
}

inline uint64_t IsPo2(uint64_t x) { return x == NearestPo2(x); }

// ceil(x/y)
template <typename X, typename Y>
inline constexpr auto RoundUp(X x, Y y) -> decltype((x + y - 1) / y) {
  return (x + y - 1) / y;
}

inline constexpr size_t Align(size_t count, size_t alignment) {
  return ((count + alignment - 1) / alignment) * alignment;
}

inline int64_t Sign(int64_t a) { return (a == 0 ? 0 : (a < 0 ? -1 : 1)); }

}  // namespace math
}  // namespace tile
}  // namespace vertexai
