#pragma once

#include <cstdint>

namespace fnv1a64 {

constexpr std::uint64_t prime = 0x100000001B3ull;
constexpr std::uint64_t basis = 0xCBF29CE484222325ull;

// compute the hash of a string literal at compile time
constexpr std::uint64_t hashlit(char const* str, std::uint64_t prev = basis) {
  return *str ? hashlit(str + 1, (*str ^ prev) * prime) : prev;
}

// compute the hash of a null-terminated string at run time
static std::uint64_t hash(char const* str) {
  std::uint64_t ret = basis;
  while (*str) {
    ret ^= *str;
    ret *= prime;
    str++;
  }
  return ret;
}

}  // namespace fnv1a64
