#include "pmlc/tools/pmlc-vulkan-runner/VulkanRuntimeTests.h"

#include "mlir/Dialect/SPIRV/SPIRVDialect.h"

using namespace pmlc::vulkan;  // NOLINT[build/namespaces]

TEST_F(RuntimeTest, SimpleTest) {
  // SPIRV module embedded into the string.
  // This module contains 4 resource variables devided into 2 sets.

  mlir::registerDialect<mlir::spirv::SPIRVDialect>();

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
