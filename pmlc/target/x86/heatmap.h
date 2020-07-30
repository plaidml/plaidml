// Copyright 2020 Intel Corporation

#pragma once

#include <map>
#include <tuple>

#include "llvm/ADT/ArrayRef.h"

#include "pmlc/dialect/pxa/transforms/passes.h"

namespace pmlc::target::x86 {

dialect::pxa::StencilCost heatmapCost(llvm::ArrayRef<int64_t> tile);

} // namespace pmlc::target::x86
