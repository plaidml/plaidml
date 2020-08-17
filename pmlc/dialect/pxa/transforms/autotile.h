// Copyright 2020 Intel Corporation

#pragma once

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Pass/Pass.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/tile.h"

namespace pmlc::dialect::pxa {

// A tile size generator is a functor from range -> list of sizes

// Produces all powers of 2 <= range
struct PowerOfTwoGenerator {
  std::vector<int64_t> operator()(int64_t range);
};

// Produces all divisors of range (including the full range)
struct EvenTilingGenerator {
  std::vector<int64_t> operator()(int64_t range);
};

// Produces only 'range' itself
struct ExactRangeGenerator {
  std::vector<int64_t> operator()(int64_t range) { return {range}; }
};

// Produces only a specific value
struct FixedTileSizeGenerator {
  explicit FixedTileSizeGenerator(int64_t val) : val(val) {}
  std::vector<int64_t> operator()(int64_t range) const { return {val}; }

  int64_t val;
};

// Produces the union of multiple generators
template <typename Head, typename... Rest>
class UnionGenerator {
public:
  explicit UnionGenerator(Head &&head, const Rest &&... rest)
      : head(std::forward<Head>(head)), rest(std::forward<Rest...>(rest...)) {}
  std::vector<int64_t> operator()(int64_t range) const {
    auto v1 = head(range);
    auto v2 = rest(range);
    std::vector<int64_t> vout;
    std::set_union(v1.begin(), v1.end(), v2.begin(), v2.end(),
                   std::back_inserter(vout));
    return vout;
  }

private:
  Head head;
  UnionGenerator<Rest...> rest;
};

template <typename Single>
class UnionGenerator<Single> {
public:
  explicit UnionGenerator(Single &&single)
      : single(std::forward<Single>(single)) {}
  std::vector<int64_t> operator()(int64_t range) const { return single(range); }

private:
  Single single;
};

// A tile cost model is a functor from an array of tile sizes (i.e.
// ArrayRef<int64_t>) to a double, which is 'inf' for infeasible tilings.  For
// example:

inline double DummyCostModel(ArrayRef<int64_t> tile, double bestCost) { return 1.0; }

// Given a generator and cost model, find the best tile size, return empty
// tiling when all tiles are infeasible
template <typename Generator, typename CostModel>
llvm::SmallVector<int64_t, 8> findBestTileSize(const Generator &generator,
                                               const CostModel &costModel,
                                               ArrayRef<int64_t> ranges) {
  // Build a list of potential tile sizes for each dimension.
  // Basically, we are caching the output of the generator in case it is
  // expensive.
  std::vector<std::vector<int64_t>> allowedTileSizes;
  for (int64_t range : ranges) {
    allowedTileSizes.emplace_back(generator(range));
  }
  // Initialize cost + tile sizes
  double bestCost = std::numeric_limits<float>::infinity();
  llvm::SmallVector<int64_t, 8> bestTileSize;
  llvm::SmallVector<int64_t, 8> curTileSize(ranges.size());
  // Build a recursive lambda to walk over the options (thanks c++14!)
  auto recurse = [&](auto &self, size_t idx) -> void {
    if (idx == allowedTileSizes.size()) {
      double newCost = costModel(curTileSize, bestCost);
      if (newCost < bestCost) {
        bestCost = newCost;
        bestTileSize = curTileSize;
      }
    } else {
      for (int64_t tileSize : allowedTileSizes[idx]) {
        curTileSize[idx] = tileSize;
        self(self, idx + 1);
      }
    }
  };
  // Call the recursive lambda
  recurse(recurse, 0);
  // Return the final result
  return bestTileSize;
}

} // namespace pmlc::dialect::pxa
