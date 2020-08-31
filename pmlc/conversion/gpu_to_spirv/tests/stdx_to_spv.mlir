// RUN: pmlc-opt -gpu-to-spirv-custom %s | FileCheck %s

module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader, Groups, Int64, Int16, Int8, Float64, Float16], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @cosh(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
    %c1 = constant 1 : index
    %c3 = constant 3 : index
    "gpu.launch_func"(%c1, %c1, %c1, %c3, %c3, %c1, %arg0, %arg1) {kernel = @cosh_kernel::@cosh_kernel} : (index, index, index, index, index, index, memref<3x3xf32>, memref<3x3xf32>) -> ()
    "gpu.launch_func"(%c1, %c1, %c1, %c3, %c3, %c1, %arg0, %arg1) {kernel = @sinh_kernel::@sinh_kernel} : (index, index, index, index, index, index, memref<3x3xf32>, memref<3x3xf32>) -> ()
    "gpu.launch_func"(%c1, %c1, %c1, %c3, %c3, %c1, %arg0, %arg1) {kernel = @floor_kernel::@floor_kernel} : (index, index, index, index, index, index, memref<3x3xf32>, memref<3x3xf32>) -> ()
    "gpu.launch_func"(%c1, %c1, %c1, %c3, %c3, %c1, %arg0, %arg1) {kernel = @tan_kernel::@tan_kernel} : (index, index, index, index, index, index, memref<3x3xf32>, memref<3x3xf32>) -> ()
    return
  }
  gpu.module @cosh_kernel {
    gpu.func @cosh_kernel(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) kernel attributes {spv.entry_point_abi = {local_size = dense<[3, 3, 1]> : vector<3xi32>}} {
      %0 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "y"} : () -> index
      %2 = load %arg0[%0, %1] : memref<3x3xf32>
      // CHECK: spv.GLSL.Exp
      // CHECK-NEXT: spv.FNegate
      // CHECK-NEXT: spv.GLSL.Exp
      // CHECK-NEXT: spv.FAdd
      // CHECK-NEXT: spv.constant
      // CHECK-NEXT: spv.FDiv
      %3 = stdx.cosh(%2) : (f32) -> f32
      store %3, %arg1[%0, %1] : memref<3x3xf32>
      gpu.return
    }
  }
  gpu.module @sinh_kernel {
    gpu.func @sinh_kernel(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) kernel attributes {spv.entry_point_abi = {local_size = dense<[3, 3, 1]> : vector<3xi32>}} {
      %0 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "y"} : () -> index
      %2 = load %arg0[%0, %1] : memref<3x3xf32>
      // CHECK: spv.GLSL.Exp
      // CHECK-NEXT: spv.FNegate
      // CHECK-NEXT: spv.GLSL.Exp
      // CHECK-NEXT: spv.FSub
      // CHECK-NEXT: spv.constant
      // CHECK-NEXT: spv.FDiv
      %3 = stdx.sinh(%2) : (f32) -> f32
      store %3, %arg1[%0, %1] : memref<3x3xf32>
      gpu.return
    }
  }
  gpu.module @floor_kernel {
    gpu.func @floor_kernel(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) kernel attributes {spv.entry_point_abi = {local_size = dense<[3, 3, 1]> : vector<3xi32>}} {
      %0 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "y"} : () -> index
      %2 = load %arg0[%0, %1] : memref<3x3xf32>
      // CHECK: spv.GLSL.Floor
      %3 = stdx.floor(%2) : (f32) -> f32
      store %3, %arg1[%0, %1] : memref<3x3xf32>
      gpu.return
    }
  }
  gpu.module @tan_kernel {
    gpu.func @tan_kernel(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) kernel attributes {spv.entry_point_abi = {local_size = dense<[3, 3, 1]> : vector<3xi32>}} {
      %0 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "y"} : () -> index
      %2 = load %arg0[%0, %1] : memref<3x3xf32>
      // CHECK: spv.GLSL.Tan
      %3 = stdx.tan(%2) : (f32) -> f32
      store %3, %arg1[%0, %1] : memref<3x3xf32>
      gpu.return
    }
  }
}
