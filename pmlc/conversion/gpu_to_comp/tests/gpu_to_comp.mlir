// RUN: pmlc-opt -pmlc-convert-gpu-to-comp %s | FileCheck %s -check-prefix CHECK -check-prefix SPACE0
// RUN: pmlc-opt -pmlc-convert-gpu-to-comp="comp-execenv-memory-space=11" %s | FileCheck %s -check-prefix CHECK -check-prefix SPACE11

module attributes {gpu.container_module} {
  //  CHECK-LABEL: func @one_gpu_func
  //   CHECK-SAME:     %[[ARG:[a-zA-Z0-9]*]]:
  //        CHECK:   %[[ENV:.*]] = comp.create_execenv
  //      SPACE11:   %[[MEM:.*]] = comp.alloc %[[ENV]] %[[ARG]]
  //        CHECK:   %[[FEV:.*]] = "comp.schedule_func"(%[[ENV]])
  //   CHECK-NEXT:     gpu.launch_func
  //  SPACE0-SAME:       %[[ARG]]
  // SPACE11-SAME:       %[[MEM]]
  //       SPACE0:   comp.wait %[[FEV]]
  //      SPACE11:   %[[REV:.*]] = comp.schedule_read %[[ARG]] from %[[MEM]] on %[[ENV]] wait for %[[FEV]]
  //      SPACE11:   comp.wait %[[REV]]
  //      SPACE11:   comp.dealloc %[[ENV]] %[[MEM]]
  //        CHECK:   comp.destroy_execenv %[[ENV]]
  func @one_gpu_func(%arg0: memref<8x32xf32>) {
    %c8 = constant 8 : index
    %c32 = constant 32 : index
    %c1 = constant 1 : index
    "gpu.launch_func"(%c8, %c1, %c1, %c32, %c1, %c1, %arg0) {kernel = @gpu_module::@zero} : (index, index, index, index, index, index, memref<8x32xf32>) -> ()
    return
  }
  gpu.module @gpu_module {
    //  CHECK-LABEL: gpu.func @zero
    //  SPACE0-SAME:     memref<8x32xf32>
    // SPACE11-SAME:     memref<8x32xf32, 11>
    gpu.func @zero(%arg0: memref<8x32xf32>) kernel attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
      %cst = constant 0.000000e+00 : f32
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      store %cst, %arg0[%0, %1] : memref<8x32xf32>
      gpu.return
    }
  }
}

module attributes {gpu.container_module} {
  //  CHECK-LABEL: func @two_gpu_func
  //   CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]*]]:
  //   CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]*]]:
  //   CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]*]]:

  //        CHECK:   %[[ENV0:.*]] = comp.create_execenv
  //      SPACE11:   %[[MEM02:.*]] = comp.alloc %[[ENV0]] %[[ARG2]]
  //        CHECK:   %[[FEV0:.*]] = "comp.schedule_func"(%[[ENV0]])
  //        CHECK:     gpu.launch_func
  //  SPACE0-SAME:       %[[ARG2]]
  // SPACE11-SAME:       %[[MEM02]]
  //       SPACE0:   comp.wait %[[FEV0]]
  //      SPACE11:   %[[REV02:.*]] = comp.schedule_read %[[ARG2]] from %[[MEM02]] on %[[ENV0]] wait for %[[FEV0]]
  //      SPACE11:   comp.wait %[[REV02]]
  //      SPACE11:   comp.dealloc %[[ENV0]] %[[MEM02]]
  //        CHECK:   comp.destroy_execenv %[[ENV0]]

  //        CHECK:   %[[ENV1:.*]] = comp.create_execenv
  //      SPACE11:   %[[MEM10:.*]] = comp.alloc %[[ENV1]] %[[ARG0]]
  //      SPACE11:   %[[MEM11:.*]] = comp.alloc %[[ENV1]] %[[ARG1]]
  //      SPACE11:   %[[MEM12:.*]] = comp.alloc %[[ENV1]] %[[ARG2]]
  //        CHECK:   %[[FEV1:.*]] = "comp.schedule_func"(%[[ENV1]])
  //        CHECK:     gpu.launch_func
  //  SPACE0-SAME:       %[[ARG0]], %[[ARG1]], %[[ARG2]]
  // SPACE11-SAME:       %[[MEM10]], %[[MEM11]], %[[MEM12]]
  //       SPACE0:   comp.wait %[[FEV1]]
  //      SPACE11:   %[[REV10:.*]] = comp.schedule_read %[[ARG0]] from %[[MEM10]] on %[[ENV1]] wait for %[[FEV1]]
  //      SPACE11:   %[[REV11:.*]] = comp.schedule_read %[[ARG1]] from %[[MEM11]] on %[[ENV1]] wait for %[[FEV1]]
  //      SPACE11:   %[[REV12:.*]] = comp.schedule_read %[[ARG2]] from %[[MEM12]] on %[[ENV1]] wait for %[[FEV1]]
  //      SPACE11:   comp.wait %[[REV10]], %[[REV11]], %[[REV12]]
  //      SPACE11:   comp.dealloc %[[ENV1]] %[[MEM10]]
  //      SPACE11:   comp.dealloc %[[ENV1]] %[[MEM11]]
  //      SPACE11:   comp.dealloc %[[ENV1]] %[[MEM12]]
  //        CHECK:   comp.destroy_execenv %[[ENV1]]
  func @two_gpu_func(%arg0: memref<8x16xf32>, %arg1: memref<16x32xf32>, %arg2: memref<8x32xf32>) {
    %c8 = constant 8 : index
    %c32 = constant 32 : index
    %c1 = constant 1 : index
    "gpu.launch_func"(%c8, %c1, %c1, %c32, %c1, %c1, %arg2) {kernel = @gpu_module::@zero} : (index, index, index, index, index, index, memref<8x32xf32>) -> ()
    "gpu.launch_func"(%c8, %c1, %c1, %c32, %c1, %c1, %arg0, %arg1, %arg2) {kernel = @gpu_module::@dot} : (index, index, index, index, index, index, memref<8x16xf32>, memref<16x32xf32>, memref<8x32xf32>) -> ()
    return
  }
  gpu.module @gpu_module {
    //  CHECK-LABEL: gpu.func @zero
    //  SPACE0-SAME:     memref<8x32xf32>
    // SPACE11-SAME:     memref<8x32xf32, 11>
    gpu.func @zero(%arg0: memref<8x32xf32>) kernel attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
      %cst = constant 0.000000e+00 : f32
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      store %cst, %arg0[%0, %1] : memref<8x32xf32>
      gpu.return
    }
    //  CHECK-LABEL: gpu.func @dot
    //  SPACE0-SAME:    memref<8x16xf32>
    //  SPACE0-SAME:    memref<16x32xf32>
    //  SPACE0-SAME:    memref<8x32xf32>
    // SPACE11-SAME:    memref<8x16xf32, 11>
    // SPACE11-SAME:    memref<16x32xf32, 11>
    // SPACE11-SAME:    memref<8x32xf32, 11>
    gpu.func @dot(%src0: memref<8x16xf32>, %src1: memref<16x32xf32>, %dst: memref<8x32xf32>) kernel attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
      %c0 = constant 0 : index
      %c16 = constant 16 : index
      %c1 = constant 1 : index
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      scf.for %arg3 = %c0 to %c16 step %c1 {
        %2 = load %src0[%0, %arg3] : memref<8x16xf32>
        %3 = load %src1[%arg3, %1] : memref<16x32xf32>
        %4 = mulf %2, %3 : f32
        %5 = load %dst[%0, %1] : memref<8x32xf32>
        %6 = addf %5, %4 : f32
        store %6, %dst[%0, %1] : memref<8x32xf32>
      }
      gpu.return
    }
  }
}
