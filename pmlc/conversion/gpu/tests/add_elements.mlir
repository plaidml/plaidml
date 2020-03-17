// RUN: pmlc-opt -legalize-gpu %s | FileCheck %s

module attributes {gpu.container_module} {
  func @get_value(%arg0: memref<3x3xf32> {tile.name = "anon"}, %arg1: memref<3x3xf32> {tile.name = "anon_0"}, %arg2: memref<3x3xf32>) {
    %c3 = constant 3 : index
    %c1 = constant 1 : index
    "gpu.launch_func"(%c3, %c1, %c1, %c3, %c1, %c1, %arg1, %arg0, %arg2) {kernel = "get_value_kernel", kernel_module = @get_value_kernel} : (index, index, index, index, index, index, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
    return
  }
  gpu.module @get_value_kernel {
    gpu.func @get_value_kernel(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = load %arg0[%0, %1] : memref<3x3xf32>
      %3 = load %arg1[%0, %1] : memref<3x3xf32>
      %4 = addf %2, %3 : f32
      store %4, %arg2[%0, %1] : memref<3x3xf32>
      gpu.return
    }
  }
}

// CHECK: attributes {spv.entry_point_abi = {local_size = dense<[3, 1, 1]> : vector<3xi32>}}
