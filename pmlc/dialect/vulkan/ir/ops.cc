// Copyright 2020 Intel Corporation

#include "pmlc/dialect/vulkan/ir/ops.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::vulkan {

VulkanDialect::VulkanDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/vulkan/ir/ops.cc.inc"
      >();
}

} // namespace pmlc::dialect::vulkan
