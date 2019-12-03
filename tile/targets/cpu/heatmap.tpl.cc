// Copyright 2018, Intel Corporation

#include "tile/targets/cpu/heatmap.h"

namespace tile::targets::cpu {

std::map<HeatmapKey, double> kHeatmap = {
    // clang-format off
  {{#data}}
    { { {{M}}, {{N}}, {{K}} }, {{GFLOPS}} },
  {{/data}}
    // clang-format on
};

}  // namespace tile::targets::cpu
