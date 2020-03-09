// Copyright 2018, Intel Corporation

#include "pmlc/dialect/pxa/transforms/heatmap.h"

namespace vertexai::tile::targets::cpu {

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

} // namespace vertexai::tile::targets::cpu
