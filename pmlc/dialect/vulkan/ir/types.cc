// Copyright 2019, Intel Corporation

#include "pmlc/dialect/vulkan/ir/types.h"

namespace pmlc::dialect::vulkan {

BufferType BufferType::get(mlir::MLIRContext *context) {
  return Base::get(context, TypeKinds::BufferType);
}

ShaderModuleType ShaderModuleType::get(mlir::MLIRContext *context) {
  return Base::get(context, TypeKinds::ShaderModuleType);
}

} // namespace pmlc::dialect::vulkan
