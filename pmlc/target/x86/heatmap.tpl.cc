// Copyright 2018, Intel Corporation
#include <cassert>

#include "pmlc/target/x86/heatmap.h"

namespace pmlc::target::x86 {

static const unsigned STARTUP_COST = 32;
static const unsigned SPEC_STENCIL_TUPLES = 1;
static const unsigned SPEC_STENCIL_INDICES = 3;
// clang-format off
// Need the spaces between braces, so the macro replacement logic doesnt trigger.
static unsigned special_stencils[1][3] = { {64, 16, 3} };
// clang-format on
uint64_t kHeatmapSize = {{#size}} {{SIZE}} {{ / size}};

uint16_t kHeatmapKeys[][3] = {
    // clang-format off
  {{#key}}
    { {{M}}, {{N}}, {{K}} },
  {{/key}}
    // clang-format on
};

float kHeatmapValues[] = {
    // clang-format off
  {{#value}}
    {{GFLOPS}},
  {{/value}}
    // clang-format on
};

static std::map<std::tuple<unsigned, unsigned, unsigned>, double> *
InitializeEfficiencyMap() {
  std::map<std::tuple<unsigned, unsigned, unsigned>, double> *ret =
      new std::map<std::tuple<unsigned, unsigned, unsigned>, double>();
  for (unsigned i = 0; i < kHeatmapSize; ++i) {
    ret->emplace(std::make_tuple(kHeatmapKeys[i][0], kHeatmapKeys[i][1],
                                 kHeatmapKeys[i][2]),
                 kHeatmapValues[i]);
  }
  return ret;
}

// Efficiency heatmap
static std::map<std::tuple<unsigned, unsigned, unsigned>, double> *heatmapPtr =
    InitializeEfficiencyMap();

std::pair<double, unsigned> HeatmapCoster(
    const unsigned *ranges,
    const unsigned count) { // ranges is always in order m, n, k, count is 3
  assert(count == 3);

  auto iter =
      heatmapPtr->find(std::make_tuple(ranges[0], ranges[1], ranges[2]));
  if (iter != heatmapPtr->end()) {
    return std::make_pair(iter->second, STARTUP_COST);
  }
  // We mainly care about M and K. If both (m, n - 1, k) and (m, n + 1, k)
  // exist, we may use their average value for prediction
  auto iter0 =
      heatmapPtr->find(std::make_tuple(ranges[0], ranges[1] - 1, ranges[2]));
  if (ranges[1] == 1 || iter0 != heatmapPtr->end()) {
    auto iter1 =
        heatmapPtr->find(std::make_tuple(ranges[0], ranges[1] + 1, ranges[2]));
    if (iter1 != heatmapPtr->end()) {
      return std::make_pair((ranges[1] > 1)
                                ? ((iter0->second + iter1->second) / 2)
                                : iter1->second,
                            STARTUP_COST);
    }
  }
  // If we cannot find (m, n, k) in the heatmap, try the special cases
  for (unsigned i = 0; i < SPEC_STENCIL_TUPLES; i++) {
    bool match = true;
    for (unsigned j = 0; j < SPEC_STENCIL_INDICES; j++) {
      if (special_stencils[i][0] != ranges[0]) {
        match = false;
        break;
      } else if (special_stencils[i][1] != ranges[1]) {
        match = false;
        break;
      } else if (special_stencils[i][2] != ranges[2]) {
        match = false;
        break;
      }
    }

    if (match) {
      return std::make_pair(0.001, STARTUP_COST);
    }
  }

  return std::make_pair(0.0, 0);
}
} // namespace pmlc::target::x86
