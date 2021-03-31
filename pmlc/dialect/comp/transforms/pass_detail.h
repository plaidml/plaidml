// Copyright 2020 Intel Corporation
#pragma once

#include "mlir/Pass/Pass.h"

namespace pmlc::dialect::comp {

#define GEN_PASS_CLASSES
#include "pmlc/dialect/comp/transforms/passes.h.inc"

} // namespace pmlc::dialect::comp
