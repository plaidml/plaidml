#pragma once

#include <memory>
#include <random>
#include <string>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Dialect/SPIRV/SPIRVBinaryUtils.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "pmlc/tools/pmlc-vulkan-runner/VulkanRuntime.h"

using namespace mlir;  // NOLINT[build/namespaces]
using namespace llvm;  // NOLINT[build/namespaces]

namespace pmlc::vulkan {
class RuntimeTest {
 public:
  mlir::LogicalResult parseAndRunModule(llvm::StringRef sourceFile, NumWorkGroups numWorkGroups);

  std::unique_ptr<float[]> createResourceVarFloat(uint32_t descriptorSet, uint32_t binding, uint32_t elementCount);

  void destroyResourceVarFloat(VulkanHostMemoryBuffer& hostMemoryBuffer);

  VulkanHostMemoryBuffer FMul(VulkanHostMemoryBuffer& var1, VulkanHostMemoryBuffer& var2);

  VulkanHostMemoryBuffer FAdd(VulkanHostMemoryBuffer& var1, VulkanHostMemoryBuffer& var2);

  bool isEqualFloat(const VulkanHostMemoryBuffer& hostMemoryBuffer1, const VulkanHostMemoryBuffer& hostMemoryBuffer2);

  llvm::DenseMap<DescriptorSetIndex, llvm::DenseMap<BindingIndex, VulkanHostMemoryBuffer>> vars;
  std::random_device randomDevice;
};

LogicalResult RuntimeTest::parseAndRunModule(llvm::StringRef sourceFile, NumWorkGroups numWorkGroups) {
  std::string errorMessage;
  auto inputFile = llvm::MemoryBuffer::getMemBuffer(sourceFile);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(inputFile), SMLoc());

  MLIRContext context;
  OwningModuleRef moduleRef(parseSourceFile(sourceMgr, &context));
  if (!moduleRef) {
    llvm::errs() << "\ncannot parse the file as a MLIR module" << '\n';
    return failure();
  }

  if (failed(runOnVulkan(moduleRef.get(), vars, numWorkGroups))) {
    return failure();
  }

  return success();
}

std::unique_ptr<float[]> RuntimeTest::createResourceVarFloat(uint32_t descriptorSet, uint32_t binding,
                                                             uint32_t elementCount) {
  std::unique_ptr<float[]> ptr(new float[elementCount]);
  std::mt19937 gen(randomDevice());
  std::uniform_real_distribution<> distribution(0.0, 10.0);
  for (uint32_t i = 0; i < elementCount; ++i) {
    ptr[i] = static_cast<float>(distribution(gen));
  }
  VulkanHostMemoryBuffer hostMemoryBuffer;
  hostMemoryBuffer.ptr = ptr.get();
  hostMemoryBuffer.size = sizeof(float) * elementCount;
  vars[descriptorSet][binding] = hostMemoryBuffer;
  return ptr;
}

void RuntimeTest::destroyResourceVarFloat(VulkanHostMemoryBuffer& hostMemoryBuffer) {
  float* ptr = static_cast<float*>(hostMemoryBuffer.ptr);
  delete ptr;
}

VulkanHostMemoryBuffer RuntimeTest::FMul(VulkanHostMemoryBuffer& var1, VulkanHostMemoryBuffer& var2) {
  VulkanHostMemoryBuffer resultHostMemoryBuffer;
  uint32_t size = var1.size / sizeof(float);
  float* result = new float[size];
  const float* rhs = reinterpret_cast<float*>(var1.ptr);
  const float* lhs = reinterpret_cast<float*>(var2.ptr);

  for (uint32_t i = 0; i < size; ++i) {
    result[i] = lhs[i] * rhs[i];
  }
  resultHostMemoryBuffer.ptr = static_cast<void*>(result);
  resultHostMemoryBuffer.size = size * sizeof(float);
  return resultHostMemoryBuffer;
}

VulkanHostMemoryBuffer RuntimeTest::FAdd(VulkanHostMemoryBuffer& var1, VulkanHostMemoryBuffer& var2) {
  VulkanHostMemoryBuffer resultHostMemoryBuffer;
  uint32_t size = var1.size / sizeof(float);
  float* result = new float[size];
  const float* rhs = reinterpret_cast<float*>(var1.ptr);
  const float* lhs = reinterpret_cast<float*>(var2.ptr);

  for (uint32_t i = 0; i < size; ++i) {
    result[i] = lhs[i] + rhs[i];
  }
  resultHostMemoryBuffer.ptr = static_cast<void*>(result);
  resultHostMemoryBuffer.size = size * sizeof(float);
  return resultHostMemoryBuffer;
}

bool RuntimeTest::isEqualFloat(const VulkanHostMemoryBuffer& hostMemoryBuffer1,
                               const VulkanHostMemoryBuffer& hostMemoryBuffer2) {
  if (hostMemoryBuffer1.size != hostMemoryBuffer2.size) return false;

  uint32_t size = hostMemoryBuffer1.size / sizeof(float);

  const float* lhs = static_cast<float*>(hostMemoryBuffer1.ptr);
  const float* rhs = static_cast<float*>(hostMemoryBuffer2.ptr);
  const float epsilon = 0.0001f;
  for (uint32_t i = 0; i < size; ++i) {
    if (fabs(lhs[i] - rhs[i]) > epsilon) return false;
  }
  return true;
}
}  // namespace pmlc::vulkan
