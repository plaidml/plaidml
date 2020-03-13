// Copyright 2018, Intel Corporation

#include "pmlc/target/x86/heatmap.h"

namespace pmlc::target::x86 {

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

} // namespace pmlc::target::x86
