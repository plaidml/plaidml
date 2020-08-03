// RUN: pmlc-opt -pmlc-convert-gpu-to-comp="comp-execenv-memory-space=11" %s | FileCheck %s
module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader, Int64, Int16, Int8, Float64, Float16], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @dot(%arg0: memref<16x32xf32>, %arg1: memref<8x16xf32>, %arg2: memref<8x32xf32>) {
    %c8 = constant 8 : index
    %c32 = constant 32 : index
    %c1 = constant 1 : index
    "gpu.launch_func"(%c8, %c1, %c1, %c32, %c1, %c1, %arg2) {kernel = @dot_kernel::@dot_kernel} : (index, index, index, index, index, index, memref<8x32xf32>) -> ()
    // CHECK: comp.create_execenv
    // CHECK: comp.alloc {{.*}} -> memref<8x32xf32, 11>
    // CHECK: comp.schedule_func
    // CHECK: comp.schedule_read
    // CHECK: comp.wait
    // CHECK: comp.dealloc
    // CHECK: comp.destroy_execenv
    "gpu.launch_func"(%c8, %c1, %c1, %c32, %c1, %c1, %arg1, %arg0, %arg2) {kernel = @dot_kernel_0::@dot_kernel} : (index, index, index, index, index, index, memref<8x16xf32>, memref<16x32xf32>, memref<8x32xf32>) -> ()
    // CHECK: comp.create_execenv
    // CHECK: comp.alloc {{.*}} -> memref<8x16xf32, 11>
    // CHECK: comp.alloc {{.*}} -> memref<16x32xf32, 11>
    // CHECK: comp.alloc {{.*}} -> memref<8x32xf32, 11>
    // CHECK: comp.schedule_func
    // CHECK: comp.schedule_read
    // CHECK: comp.schedule_read
    // CHECK: comp.schedule_read
    // CHECK: comp.wait
    // CHECK: comp.dealloc
    // CHECK: comp.dealloc
    // CHECK: comp.dealloc
    // CHECK: comp.destroy_execenv
    return
  }
  gpu.module @dot_kernel {
    gpu.func @dot_kernel(%arg0: memref<8x32xf32>) kernel attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
    // CHECK: gpu.func @dot_kernel(%arg0: memref<8x32xf32, 11>)
      %cst = constant 0.000000e+00 : f32
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      store %cst, %arg0[%0, %1] : memref<8x32xf32>
      gpu.return
    }
  }
  gpu.module @dot_kernel_0 {
    gpu.func @dot_kernel(%arg0: memref<8x16xf32>, %arg1: memref<16x32xf32>, %arg2: memref<8x32xf32>) kernel attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
      // CHECK: gpu.func @dot_kernel(%arg0: memref<8x16xf32, 11>, %arg1: memref<16x32xf32, 11>, %arg2: memref<8x32xf32, 11>)
      %c0 = constant 0 : index
      %c16 = constant 16 : index
      %c1 = constant 1 : index
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      scf.for %arg3 = %c0 to %c16 step %c1 {
        %2 = load %arg0[%0, %arg3] : memref<8x16xf32>
        %3 = load %arg1[%arg3, %1] : memref<16x32xf32>
        %4 = mulf %2, %3 : f32
        %5 = load %arg2[%0, %1] : memref<8x32xf32>
        %6 = addf %5, %4 : f32
        store %6, %arg2[%0, %1] : memref<8x32xf32>
      }
      gpu.return
    }
  }
}
