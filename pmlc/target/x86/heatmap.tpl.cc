// Copyright 2020 Intel Corporation

#include <cstdint>

namespace pmlc::target::x86 {

// clang-format off
uint64_t kHeatmapSize = {{#size}} {{SIZE}} {{ / size}};
// clang-format on

// Tiles in { N, M, K } format
uint16_t kHeatmapKeys[][3] = {
    // clang-format off
  {{#key}}
    { {{N}}, {{M}}, {{K}} },
  {{/key}}
    // clang-format on
};

// Throughput in GFLOPS
float kHeatmapValues[] = {
    // clang-format off
  {{#value}}
    {{GFLOPS}},
  {{/value}}
    // clang-format on
};

} // namespace pmlc::target::x86
