#pragma once

namespace vertexai {
namespace tile {
namespace lang {

inline uint64_t NearestPo2(uint64_t x) {
  uint64_t po2 = 1;
  while (po2 < x) {
    po2 *= 2;
  }
  return po2;
}

inline uint64_t IsPo2(uint64_t x) { return x == NearestPo2(x); }

// ceil(x/y)
inline uint64_t RoundUp(uint64_t x, uint64_t y) { return (x + y - 1) / y; }

inline int64_t Sign(int64_t a) { return (a == 0 ? 0 : (a < 0 ? -1 : 1)); }

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
