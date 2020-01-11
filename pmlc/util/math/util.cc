
#include "pmlc/util/math/util.h"

#include <algorithm>

namespace vertexai {
namespace tile {
namespace math {

Seive::Seive(uint64_t size) : first_factor(size) {
  // Intialize each element to be '2' or 'prime' initially
  primes.push_back(2);
  for (uint64_t i = 1; i < size; i++) {
    first_factor[i] = i % 2 == 0 ? 2 : i;
  }
  // Now do the Seive on odds
  for (uint64_t i = 3; i < size; i += 2) {
    if (first_factor[i] != i) {
      continue;
    }  // Skip non-prime
    primes.push_back(i);
    for (uint64_t j = 3 * i; j < size; j += 2 * i) {
      first_factor[j] = std::min(first_factor[j], i);
    }
  }
}

uint64_t FirstFactor(uint64_t in) {
  uint64_t k_seive_size = 65537;
  static Seive seive(k_seive_size);
  if (in < k_seive_size) {
    return seive.first_factor[in];
  }
  // Fallback code >= seive_size
  for (uint64_t p : seive.primes) {
    if (in % p == 0) {
      return p;
    }
    if (p * p > in) {
      return in;
    }
  }
  // Very fallback code >= seive_size^2
  for (uint64_t p = k_seive_size; p * p <= in; p += 2) {
    if (in % p == 0) {
      return p;
    }
  }
  return in;
}

std::vector<uint64_t> Factor(uint64_t in) {
  std::vector<uint64_t> out;
  // printf("%llu: ", in);
  while (in != 1) {
    uint64_t p = FirstFactor(in);
    // printf("%llu, ", p);
    out.push_back(p);
    in /= p;
  }
  // printf("\n");
  return out;
}

uint64_t NumFactors(uint64_t in) {
  uint64_t count = 0;
  while (in != 1) {
    in /= FirstFactor(in);
    count++;
  }
  return count;
}

bool IsPrime(uint64_t in) { return FirstFactor(in) == in; }

}  // namespace math
}  // namespace tile
}  // namespace vertexai
