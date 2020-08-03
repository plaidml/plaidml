// Copyright 2020 Intel Corporation

#pragma once

#include <functional>
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::dialect::comp {

std::unique_ptr<mlir::Pass> createExecEnvCoalescingPass();

} // namespace pmlc::dialect::comp
