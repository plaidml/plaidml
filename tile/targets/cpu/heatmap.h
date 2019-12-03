// Copyright 2018, Intel Corporation

#pragma once

#include <map>
#include <tuple>

namespace tile::targets::cpu {

using HeatmapKey = std::tuple<unsigned, unsigned, unsigned>;

extern std::map<HeatmapKey, double> kHeatmap;

}  // namespace tile::targets::cpu
