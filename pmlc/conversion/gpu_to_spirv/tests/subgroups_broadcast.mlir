// RUN: pmlc-opt -gpu-to-spirv-custom %s | FileCheck %s

module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.5, [Shader, GroupNonUniformBallot], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @main() {
    %c3 = constant 3 : index
    %c1 = constant 1 : index
    %2 = alloc() : memref<3x3xf32>
    "gpu.launch_func"(%c3, %c1, %c1, %c3, %c1, %c1, %2) {kernel = @bcast_kernel::@bcast_kernel} : (index, index, index, index, index, index, memref<3x3xf32>) -> ()
    return
  }
  
  gpu.module @bcast_kernel {
    gpu.func @bcast_kernel(%arg0: memref<3x3xf32>) kernel attributes {spv.entry_point_abi = {local_size = dense<[3, 1, 1]> : vector<3xi32>}} {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %c1 = constant 1 : index 
      // CHECK: spv.GroupNonUniformBroadcast "Subgroup" %{{.*}}, %{{.*}} : f32
      %4 = load %arg0[%0, %1] : memref<3x3xf32>
      %5 = stdx.subgroup_broadcast(%4, %c1) : f32
      store %5, %arg0[%0, %1] : memref<3x3xf32>
      gpu.return
    }
  }
}
