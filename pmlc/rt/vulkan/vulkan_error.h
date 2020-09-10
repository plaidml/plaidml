// Copyright 2020, Intel Corporation

#pragma once

#include <stdexcept>

#include "volk.h" // NOLINT[build/include_subdir]

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

namespace pmlc::rt::vulkan {

inline void throwOnVulkanError(VkResult result, const char *call) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error{
        llvm::formatv("{0} failed with error code {1}", call, result)};
  }
}

inline void emitVulkanError(const llvm::Twine &message, VkResult error) {
  llvm::errs()
      << message.concat(" failed with error code ").concat(llvm::Twine{error});
}

#define RETURN_ON_VULKAN_ERROR(result, msg)                                    \
  if ((result) != VK_SUCCESS) {                                                \
    emitVulkanError(msg, (result));                                            \
    return failure();                                                          \
  }

} // namespace pmlc::rt::vulkan
