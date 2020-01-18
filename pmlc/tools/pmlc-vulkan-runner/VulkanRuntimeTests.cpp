#include "pmlc/tools/pmlc-vulkan-runner/VulkanRuntimeTests.h"

using namespace pmlc::vulkan;  // NOLINT[build/namespaces]

int main() {
  // TEST_F(RuntimeTest, SimpleTest) {
  // SPIRV module embedded into the string.
  // This module contains 4 resource variables devided into 2 sets.

  RuntimeTest rt;

  std::string spirvModuleSource =
      "spv.module \"Logical\" \"GLSL450\" {\n"
      "spv.globalVariable @var3 bind(1, 1) : "
      "!spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
      "spv.globalVariable @var2 bind(1, 0) : "
      "!spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
      "spv.globalVariable @var1 bind(0, 1) : "
      "!spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
      "spv.globalVariable @var0 bind(0, 0) : "
      "!spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
      "spv.globalVariable @globalInvocationID "
      "built_in(\"GlobalInvocationId\"): !spv.ptr<vector<3xi32>, Input>\n"
      "func @kernel() -> () {\n"
      "%c0 = spv.constant 0 : i32\n"

      "%0 = spv._address_of @var0 : !spv.ptr<!spv.struct<!spv.array<1024 x f32 "
      "[4]> [0]>, StorageBuffer>\n"
      "%1 = spv._address_of @var1 : !spv.ptr<!spv.struct<!spv.array<1024 x f32 "
      "[4]> [0]>, StorageBuffer>\n"
      "%2 = spv._address_of @var2 : !spv.ptr<!spv.struct<!spv.array<1024 x f32 "
      "[4]> [0]>, StorageBuffer>\n"
      "%3 = spv._address_of @var3 : !spv.ptr<!spv.struct<!spv.array<1024 x f32 "
      "[4]> [0]>, StorageBuffer>\n"

      "%ptr_id = spv._address_of @globalInvocationID: !spv.ptr<vector<3xi32>, "
      "Input>\n"

      "%id = spv.AccessChain %ptr_id[%c0] : !spv.ptr<vector<3xi32>, Input>\n"
      "%index = spv.Load \"Input\" %id: i32\n"

      "%4 = spv.AccessChain %0[%c0, %index] : "
      "!spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
      "%5 = spv.AccessChain %1[%c0, %index] : "
      "!spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
      "%6 = spv.AccessChain %2[%c0, %index] : "
      "!spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
      "%7 = spv.AccessChain %3[%c0, %index] : "
      "!spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"

      "%8 = spv.Load \"StorageBuffer\" %4 : f32\n"
      "%9 = spv.Load \"StorageBuffer\" %5 : f32\n"
      "%10 = spv.Load \"StorageBuffer\" %6 : f32\n"

      "%11 = spv.FMul %8, %9 : f32\n"
      "%12 = spv.FAdd %11, %10 : f32\n"

      "spv.Store \"StorageBuffer\" %7, %12 : f32\n"
      "spv.Return\n"
      "}\n"
      "spv.EntryPoint \"GLCompute\" @kernel, @globalInvocationID\n"
      "spv.ExecutionMode @kernel \"LocalSize\", 1, 1, 1\n"
      "} attributes {\n"
      "capabilities = [\"Shader\"],\n"
      "extensions = [\"SPV_KHR_storage_buffer_storage_class\"]\n"
      "}\n";

  auto resOne = rt.createResourceVarFloat(0, 0, 1024);
  auto resTwo = rt.createResourceVarFloat(0, 1, 1024);
  auto resThree = rt.createResourceVarFloat(1, 0, 1024);
  auto resFour = rt.createResourceVarFloat(1, 1, 1024);

  // auto fmulResult = rt.FMul(rt.vars[0][0], rt.vars[0][1]);
  // auto expected = rt.FAdd(rt.vars[1][0], fmulResult);

  NumWorkGroups numWorkGroups;
  numWorkGroups.x = 1024;
  rt.parseAndRunModule(spirvModuleSource, numWorkGroups);

  // ASSERT_TRUE(isEqualFloat(expected, vars[1][1]));

  // destroyResourceVarFloat(fmulResult);
  // destroyResourceVarFloat(expected);
  return 0;
}
