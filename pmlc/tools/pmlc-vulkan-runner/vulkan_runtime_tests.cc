
#include "pmlc/tools/pmlc-vulkan-runner/vulkan_runtime.h"

#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <string>
#include <utility>

#include "gmock/gmock.h"

#include "mlir/Dialect/SPIRV/SPIRVBinaryUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;  // NOLINT[build/namespaces]
using namespace llvm;  // NOLINT[build/namespaces]

namespace pmlc::vulkan {
class RuntimeTest : public ::testing::Test {
 public:
  LogicalResult parseAndRunModule(llvm::StringRef sourceFile, NumWorkGroups numWorkGroups) {
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

  std::unique_ptr<float[]> createResourceVarFloat(uint32_t descriptorSet, uint32_t binding, uint32_t elementCount) {
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

  void destroyResourceVarFloat(VulkanHostMemoryBuffer& hostMemoryBuffer) {
    float* ptr = static_cast<float*>(hostMemoryBuffer.ptr);
    delete ptr;
  }

  llvm::DenseMap<DescriptorSetIndex, llvm::DenseMap<BindingIndex, VulkanHostMemoryBuffer>> vars;

 protected:
  VulkanHostMemoryBuffer FMul(VulkanHostMemoryBuffer& var1, VulkanHostMemoryBuffer& var2) {
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

  VulkanHostMemoryBuffer FAdd(VulkanHostMemoryBuffer& var1, VulkanHostMemoryBuffer& var2) {
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

  bool isEqualFloat(const VulkanHostMemoryBuffer& hostMemoryBuffer1, const VulkanHostMemoryBuffer& hostMemoryBuffer2) {
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

 protected:
  std::random_device randomDevice;
};

TEST_F(RuntimeTest, SimpleTest) {
  // SPIRV module embedded into the string.
  // This module contains 4 resource variables devided into 2 sets.

  std::string spirvModuleSource =  // NOLINT
      R"***(

  spv.module "Logical" "GLSL450" {
    spv.globalVariable @var3 bind(1, 1) : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>
    spv.globalVariable @var2 bind(1, 0) : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>
    spv.globalVariable @var1 bind(0, 1) : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>
    spv.globalVariable @var0 bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>
    spv.globalVariable @globalInvocationID built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    func @kernel() {
      %0 = spv.constant 0 : i32
      %1 = spv._address_of @var0 : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>
      %2 = spv._address_of @var1 : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>
      %3 = spv._address_of @var2 : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>
      %4 = spv._address_of @var3 : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>
      %5 = spv._address_of @globalInvocationID : !spv.ptr<vector<3xi32>, Input>
      %6 = spv.AccessChain %5[%0] : !spv.ptr<vector<3xi32>, Input>
      %7 = spv.Load "Input" %6 : i32
      %8 = spv.AccessChain %1[%0, %7] : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>
      %9 = spv.AccessChain %2[%0, %7] : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>
      %10 = spv.AccessChain %3[%0, %7] : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>
      %11 = spv.AccessChain %4[%0, %7] : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>
      %12 = spv.Load "StorageBuffer" %8 : f32
      %13 = spv.Load "StorageBuffer" %9 : f32
      %14 = spv.Load "StorageBuffer" %10 : f32
      %15 = spv.FMul %12, %13 : f32
      %16 = spv.FAdd %15, %14 : f32
      spv.Store "StorageBuffer" %11, %16 : f32
      spv.Return
    }
    spv.EntryPoint "GLCompute" @kernel, @globalInvocationID
    spv.ExecutionMode @kernel "LocalSize", 1, 1, 1
  } attributes {capabilities = ["Shader"], extensions = ["SPV_KHR_storage_buffer_storage_class"]}

  )***";                           // NOLINT

  auto resOne = createResourceVarFloat(0, 0, 1024);
  auto resTwo = createResourceVarFloat(0, 1, 1024);
  auto resThree = createResourceVarFloat(1, 0, 1024);
  auto resFour = createResourceVarFloat(1, 1, 1024);

  auto fmulResult = FMul(vars[0][0], vars[0][1]);
  auto expected = FAdd(vars[1][0], fmulResult);

  NumWorkGroups numWorkGroups;
  numWorkGroups.x = 1024;

  ASSERT_TRUE(succeeded(parseAndRunModule(spirvModuleSource, numWorkGroups)));
  ASSERT_TRUE(isEqualFloat(expected, vars[1][1]));

  destroyResourceVarFloat(fmulResult);
  destroyResourceVarFloat(expected);
}
}  // namespace pmlc::vulkan
