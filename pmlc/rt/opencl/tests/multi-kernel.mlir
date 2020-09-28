// RUN: pmlc-opencl-runner %s | FileCheck %s

// CHECK: [23,   23,   23],
// CHECK: [23,   23,   23],
// CHECK: [23,   23,   23]


module attributes {
  gpu.container_module,
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Kernel, Addresses], []>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
} {
  func @main() {
    %arg0 = alloc() : memref<3x3xf32>
    %arg1 = alloc() : memref<3x3xf32>
    %arg2 = alloc() : memref<3x3xf32>

    %c3 = constant 3 : index
    %c1 = constant 1 : index

    "gpu.launch_func"(%c3, %c1, %c1, %c3, %c1, %c1, %arg2) { kernel = @dot_kernel::@dot_kernel }
      : (index, index, index, index, index, index, memref<3x3xf32>) -> ()
    "gpu.launch_func"(%c3, %c1, %c1, %c3, %c1, %c1, %arg0) { kernel = @dot_kernel_0::@dot_kernel_0 }
      : (index, index, index, index, index, index, memref<3x3xf32>) -> ()
    "gpu.launch_func"(%c3, %c1, %c1, %c3, %c1, %c1, %arg1) { kernel = @dot_kernel_1::@dot_kernel_1 }
      : (index, index, index, index, index, index, memref<3x3xf32>) -> ()
    "gpu.launch_func"(%c3, %c1, %c1, %c3, %c1, %c1, %arg1, %arg0, %arg2) { kernel = @dot_kernel_2::@dot_kernel_2 }
      : (index, index, index, index, index, index, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()

    %arg5 = memref_cast %arg2 : memref<3x3xf32> to memref<?x?xf32>
    %arg6 = memref_cast %arg5 : memref<?x?xf32> to memref<*xf32>
    call @print_memref_f32(%arg6) : (memref<*xf32>) -> ()
    return
  }

  gpu.module @dot_kernel {
    gpu.func @dot_kernel(%arg0: memref<3x3xf32>)
    kernel attributes {spv.entry_point_abi = {local_size = dense<[3, 1, 1]> : vector<3xi32>}} {
      %cst = constant 5.000000e+00 : f32
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      store %cst, %arg0[%0, %1] : memref<3x3xf32>
      gpu.return
    }
  }

  gpu.module @dot_kernel_0 {
    gpu.func @dot_kernel_0(%arg0: memref<3x3xf32>)
    kernel attributes {spv.entry_point_abi = {local_size = dense<[3, 1, 1]> : vector<3xi32>}} {
      %cst = constant 3.000000e+00 : f32
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      store %cst, %arg0[%0, %1] : memref<3x3xf32>
      gpu.return
    }
  }

  gpu.module @dot_kernel_1 {
    gpu.func @dot_kernel_1(%arg0: memref<3x3xf32>)
    kernel attributes {spv.entry_point_abi = {local_size = dense<[3, 1, 1]> : vector<3xi32>}} {
      %cst = constant 2.000000e+00 : f32
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      store %cst, %arg0[%0, %1] : memref<3x3xf32>
      gpu.return
    }
  }

  gpu.module @dot_kernel_2 {
    gpu.func @dot_kernel_2(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>)
    kernel attributes {spv.entry_point_abi = {local_size = dense<[3, 1, 1]> : vector<3xi32>}} {
      %c0 = constant 0 : index
      %c3 = constant 3 : index
      %c1 = constant 1 : index
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      scf.for %arg3 = %c0 to %c3 step %c1 {
        %2 = load %arg0[%0, %arg3] : memref<3x3xf32>
        %3 = load %arg1[%arg3, %1] : memref<3x3xf32>
        %4 = mulf %2, %3 : f32
        %5 = load %arg2[%0, %1] : memref<3x3xf32>
        %6 = addf %5, %4 : f32
        store %6, %arg2[%0, %1] : memref<3x3xf32>
      }
      gpu.return
    }
  }

  func @print_memref_f32(%ptr : memref<*xf32>)
}
