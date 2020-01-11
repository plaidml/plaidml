#pragma once

#include <stddef.h>

#include <cstdint>
#include <vector>

namespace pmlc::util::math {

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

struct Seive {
  // First factor for each number.  We leave 0 + 1 in the list to make things easier to read
  std::vector<uint64_t> first_factor;
  std::vector<uint64_t> primes;
  explicit Seive(uint64_t size);
};

// Reeturn the first prime factor of a number
uint64_t FirstFactor(uint64_t in);

// Return the full factorization of a number
std::vector<uint64_t> Factor(uint64_t in);

// Return the number of factors a number has
uint64_t NumFactors(uint64_t in);

// Check if a number is prime
bool IsPrime(uint64_t in);

}  // namespace pmlc::util::math
