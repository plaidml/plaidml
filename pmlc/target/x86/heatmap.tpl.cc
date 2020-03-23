// Copyright 2020 Intel Corporation

#include <cstdint>

namespace pmlc::target::x86 {

// clang-format off
uint64_t kHeatmapSize = {{#size}} {{SIZE}} {{ / size}};
// clang-format on

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

} // namespace pmlc::target::x86
