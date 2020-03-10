// RUN: pmlc-vulkan-runner %s --entry-point-result=void 

module attributes {gpu.container_module} {
  func @dot(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) {
    %c3 = constant 3 : index
    %c1 = constant 1 : index
    "gpu.launch_func"(%c3, %c1, %c1, %c3, %c1, %c1, %arg1, %arg0, %arg2) {kernel = "dot_kernel", kernel_module = @dot_kernel_0} : (index, index, index, index, index, index, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
    return
  }
  gpu.module @dot_kernel_0 {
    gpu.func @dot_kernel(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) 
      attributes {gpu.kernel, spv.entry_point_abi = {local_size = dense<[3, 1, 1]>: vector<3xi32>}}{
      %c1 = constant 1 : index
      %c3 = constant 3 : index
      %c0 = constant 0 : index
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      loop.for %arg3 = %c0 to %c3 step %c1 {
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
  
  func @main() {
    %arg0 = alloc() : memref<3x3xf32>
    %arg1 = alloc() : memref<3x3xf32>
    %arg2 = alloc() : memref<3x3xf32>
    %0 = constant 0 : i32
    %1 = constant 1 : i32
    %2 = constant 2 : i32
    %arg3 = memref_cast %arg0 : memref<3x3xf32> to memref<?x?xf32>
    %arg4 = memref_cast %arg1 : memref<3x3xf32> to memref<?x?xf32>
    %arg5 = memref_cast %arg2 : memref<3x3xf32> to memref<?x?xf32>
    call @setResourceData2D(%0, %0, %arg3, %0) : (i32, i32, memref<?x?xf32>, i32) -> ()
    call @setResourceData2D(%0, %1, %arg4, %1) : (i32, i32, memref<?x?xf32>, i32) -> ()
    call @setResourceData2D(%0, %2, %arg5, %2) : (i32, i32, memref<?x?xf32>, i32) -> ()

    call @dot(%arg0, %arg1, %arg2) : ( memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
    
    %arg6 = memref_cast %arg5 : memref<?x?xf32> to memref<*xf32>
    call @print_memref_f32(%arg6) : (memref<*xf32>) -> ()
    return
  }
  func @setResourceData2D(%0 : i32, %1 : i32, %2 : memref<?x?xf32>, %4 : i32)
  func @print_memref_f32(%ptr : memref<*xf32>)
}


