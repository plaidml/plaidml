// RUN: pmlc-vulkan-runner %s | FileCheck %s

// CHECK: [0,   0,   0],
// CHECK: [1,   1,   1],
// CHECK: [2,   2,   2]


module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader, Groups], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @main() {
    %c3 = constant 3 : index
    %c1 = constant 1 : index
    %2 = alloc() : memref<3x3xf32>
    "gpu.launch_func"(%c3, %c1, %c1, %c3, %c1, %c1, %2) {kernel = @bcast_kernel::@bcast_kernel} : (index, index, index, index, index, index, memref<3x3xf32>) -> ()
    %3 = memref_cast %2 : memref<3x3xf32> to memref<?x?xf32>
    %4 = memref_cast %3 : memref<?x?xf32> to memref<*xf32>
    call @print_memref_f32(%4) : (memref<*xf32>) -> ()
    return
  }
  spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader, Groups], [SPV_KHR_storage_buffer_storage_class]> {
    spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
    spv.globalVariable @bcast_kernel_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<9 x f32, stride=4> [0]>, StorageBuffer>
    spv.func @bcast_kernel() "None" attributes {workgroup_attributions = 0 : i64} {
      %0 = spv._address_of @bcast_kernel_arg_0 : !spv.ptr<!spv.struct<!spv.array<9 x f32, stride=4> [0]>, StorageBuffer>
      %1 = spv._address_of @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
      %2 = spv.Load "Input" %1 : vector<3xi32>
      %3 = spv.CompositeExtract %2[0 : i32] : vector<3xi32>
      %4 = spv._address_of @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
      %5 = spv.Load "Input" %4 : vector<3xi32>
      %6 = spv.CompositeExtract %5[0 : i32] : vector<3xi32>
      %7 = spv.ConvertSToF %6 : i32 to f32
      %bcast7 = spv.GroupBroadcast "Subgroup" %7, %3 : f32, i32
      %8 = spv.constant 0 : i32
      %9 = spv.constant 0 : i32
      %10 = spv.constant 3 : i32
      %11 = spv.IMul %10, %3 : i32
      %12 = spv.IAdd %9, %11 : i32
      %13 = spv.constant 1 : i32
      %14 = spv.IMul %13, %6 : i32
      %15 = spv.IAdd %12, %14 : i32
      %16 = spv.AccessChain %0[%8, %15] : !spv.ptr<!spv.struct<!spv.array<9 x f32, stride=4> [0]>, StorageBuffer>, i32, i32
      spv.Store "StorageBuffer" %16, %bcast7 : f32
      spv.Return
    }
    spv.EntryPoint "GLCompute" @bcast_kernel, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
    spv.ExecutionMode @bcast_kernel "LocalSize", 3, 1, 1
  }
  gpu.module @bcast_kernel {
    gpu.func @bcast_kernel(%arg0: memref<3x3xf32>) kernel attributes {spv.entry_point_abi = {local_size = dense<[3, 1, 1]> : vector<3xi32>}} {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = index_cast %1 : index to i32
      %3 = sitofp %2 : i32 to f32
      store %3, %arg0[%0, %1] : memref<3x3xf32>
      gpu.return
    }
  }
  func @print_memref_f32(memref<*xf32>)
}