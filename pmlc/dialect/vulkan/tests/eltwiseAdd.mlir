// RUN: pmlc-opt --pmlc-convert-gpu-to-vulkan-dialect %s | FileCheck %s

module attributes {
  gpu.container_module,
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader, Int64, Int16, Int8, Float64, Float16], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
} {
  func @eltwise_add(%arg0: memref<10x20xf32>, %arg1: memref<10x20xf32>, %arg2: memref<10x20xf32>) {
    %c10 = constant 10 : index
    %c20 = constant 20 : index
    %c1 = constant 1 : index
    "gpu.launch_func"(%c10, %c1, %c1, %c20, %c1, %c1, %arg1, %arg0, %arg2) {kernel = @eltwise_add_kernel::@eltwise_add_kernel}
     : (index, index, index, index, index, index, memref<10x20xf32>, memref<10x20xf32>, memref<10x20xf32>) -> ()
    return
  }
  spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
    spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
    spv.globalVariable @eltwise_add_kernel_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<200 x f32, stride=4> [0]>, StorageBuffer>
    spv.globalVariable @eltwise_add_kernel_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<!spv.array<200 x f32, stride=4> [0]>, StorageBuffer>
    spv.globalVariable @eltwise_add_kernel_arg_2 bind(0, 2) : !spv.ptr<!spv.struct<!spv.array<200 x f32, stride=4> [0]>, StorageBuffer>
    spv.func @eltwise_add_kernel() "None" attributes {workgroup_attributions = 0 : i64} {
      %0 = spv.constant 0 : i32
      %1 = spv.constant 20 : i32
      %2 = spv._address_of @eltwise_add_kernel_arg_2 : !spv.ptr<!spv.struct<!spv.array<200 x f32, stride=4> [0]>, StorageBuffer>
      %3 = spv._address_of @eltwise_add_kernel_arg_1 : !spv.ptr<!spv.struct<!spv.array<200 x f32, stride=4> [0]>, StorageBuffer>
      %4 = spv._address_of @eltwise_add_kernel_arg_0 : !spv.ptr<!spv.struct<!spv.array<200 x f32, stride=4> [0]>, StorageBuffer>
      %5 = spv._address_of @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
      %6 = spv.Load "Input" %5 : vector<3xi32>
      %7 = spv.CompositeExtract %6[0 : i32] : vector<3xi32>
      %8 = spv._address_of @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
      %9 = spv.Load "Input" %8 : vector<3xi32>
      %10 = spv.CompositeExtract %9[0 : i32] : vector<3xi32>
      %11 = spv.IMul %7, %1 : i32
      %12 = spv.IAdd %11, %10 : i32
      %13 = spv.AccessChain %4[%0, %12] : !spv.ptr<!spv.struct<!spv.array<200 x f32, stride=4> [0]>, StorageBuffer>
      %14 = spv.Load "StorageBuffer" %13 : f32
      %15 = spv.AccessChain %3[%0, %12] : !spv.ptr<!spv.struct<!spv.array<200 x f32, stride=4> [0]>, StorageBuffer>
      %16 = spv.Load "StorageBuffer" %15 : f32
      %17 = spv.FAdd %14, %16 : f32
      %18 = spv.AccessChain %2[%0, %12] : !spv.ptr<!spv.struct<!spv.array<200 x f32, stride=4> [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %18, %17 : f32
      spv.Return
    }
    spv.EntryPoint "GLCompute" @eltwise_add_kernel, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
    spv.ExecutionMode @eltwise_add_kernel "LocalSize", 20, 1, 1
  }
  gpu.module @eltwise_add_kernel {
    gpu.func @eltwise_add_kernel(%arg0: memref<10x20xf32>, %arg1: memref<10x20xf32>, %arg2: memref<10x20xf32>) 
    kernel attributes {spv.entry_point_abi = {local_size = dense<[20, 1, 1]> : vector<3xi32>}} {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = load %arg0[%0, %1] : memref<10x20xf32>
      %3 = load %arg1[%0, %1] : memref<10x20xf32>
      %4 = addf %2, %3 : f32
      store %4, %arg2[%0, %1] : memref<10x20xf32>
      gpu.return
    }
  }

  // CHECK: "vk.createVulkanLaunchKernelAction"() {callee = @createVulkanLaunchKernelAction} : () -> ()
  // CHECK: "vk.setLaunchKernelAction"() {callee = @setVulkanLaunchKernelAction} : () -> ()
  // CHECK: "vk.AddVulkanLaunchActionToSchedule"() {callee = @addVulkanLaunchActionToSchedule} : () -> ()
  // CHECK: "vk.SubmitCommandBuffers"() {callee = @submitCommandBuffers} : () -> ()
  // CHECK: "vk.DeinitVulkan"() {callee = @deinitVulkan} : () -> ()
}
