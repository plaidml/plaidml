// Copyright 2020 Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::target::x86 {

std::unique_ptr<mlir::Pass> createTraceLinkingPass();

} // namespace pmlc::target::x86
