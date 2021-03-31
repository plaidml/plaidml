// RUN: pmlc-opt -pxa-stride-info  %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0, 0, 0)>
#map2 = affine_map<() -> (100, 100, 100)>

func @simple(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>, %arg2: memref<100x100xf32>) {
  affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) {
    %0 = pxa.load %arg1[%i, %k] : memref<100x100xf32>
    // CHECK: strides: 0:[^bb0:%arg0=100, ^bb0:%arg2=1]
    %1 = pxa.load %arg0[%k, %j] : memref<100x100xf32>
    // CHECK: strides: 0:[^bb0:%arg1=1, ^bb0:%arg2=100]
    %2 = mulf %0, %1 : f32
    pxa.reduce addf %2, %arg2[%i, %j] : memref<100x100xf32>
    // CHECK: strides: 0:[^bb0:%arg0=100, ^bb0:%arg1=1]
  }
  return
}

func @symbolic_fail(%arg0: memref<100x?xf32>) {
  %c0 = constant 0 : index
  %d1 = dim %arg0, %c0 : memref<100x?xf32>
  affine.parallel (%i, %j) = (0, 0) to (100, symbol(%d1)) {
    %0 = pxa.load %arg0[%i, %j] : memref<100x?xf32>
    // CHECK: strides: none
  }
  return
}

func @for_diagonal(%arg0: memref<100x100xf32>) {
  affine.for %i = 0 to 10 {
    %0 = pxa.load %arg0[%i, %i] : memref<100x100xf32>
    // CHECK: strides: 0:[^bb0:%arg0=101]
  }
  return
}

func @for_step(%arg0: memref<100x100xf32>) {
  affine.for %i = 0 to 10 step 2 {
    %0 = pxa.load %arg0[%i, %i] : memref<100x100xf32>
    // CHECK: strides: 0:[^bb0:%arg0=202]
  }
  return
}

func @parallel_step(%arg0: memref<100x100xf32>) {
  affine.parallel (%i, %j) = (0, 0) to (10, 10) step (2, 5) {
    %0 = pxa.load %arg0[%i, %j] : memref<100x100xf32>
    // CHECK: strides: 0:[^bb0:%arg0=200, ^bb0:%arg1=5]
  }
  return
}

func @parallel_tile(%arg0: memref<100x100xf32>) {
  affine.parallel (%i, %j) = (0, 0) to (100, 100) step (10, 10) {
    affine.parallel (%i2, %j2) = (%i, %j) to (%i + 10, %j + 10) {
      %0 = pxa.load %arg0[%i2, %j2] : memref<100x100xf32>
      // CHECK: strides: 0:[^bb0:%arg0=100, ^bb0:%arg1=1, ^bb1:%arg0=1000, ^bb1:%arg1=10]
    }
  }
  return
}

func @affine_apply(%arg0: memref<100x100xf32>) {
  affine.for %i = 0 to 10 {
    %0 = affine.apply affine_map<(d1) -> (5 * d1)>(%i)
    %1 = pxa.load %arg0[%0, %i] : memref<100x100xf32>
    // CHECK: strides: 0:[^bb0:%arg0=501]
  }
  return
}

func @affine_apply_add(%arg0: memref<100x100xf32>) {
  affine.for %i = 0 to 10 {
    %0 = affine.apply affine_map<(d1) -> (d1 + 10)>(%i)
    %1 = pxa.load %arg0[%0, %i] : memref<100x100xf32>
    // CHECK: strides: 1000:[^bb0:%arg0=101]
  }
  return
}

func @dot_tiled(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) {
  affine.parallel (%i0, %j0, %k0) = (0, 0, 0) to (8, 8, 8) step (2, 2, 2) {
    affine.parallel (%i1, %j1, %k1) = (%i0, %j0, %k0) to (%i0 + 2, %j0 + 2, %k0 + 2) {
      %0 = pxa.load %A[%i1, %k1] : memref<8x8xf32>
      // CHECK: strides: 0:[^bb0:%arg0=8, ^bb0:%arg2=1, ^bb1:%arg0=16, ^bb1:%arg2=2]
      %1 = pxa.load %B[%k1, %j1] : memref<8x8xf32>
      // CHECK: strides: 0:[^bb0:%arg1=1, ^bb0:%arg2=8, ^bb1:%arg1=2, ^bb1:%arg2=16]
      %2 = mulf %0, %1 : f32
      pxa.reduce addf %2, %C[%i1, %j1] : memref<8x8xf32>
      // CHECK: strides: 0:[^bb0:%arg0=8, ^bb0:%arg1=1, ^bb1:%arg0=16, ^bb1:%arg1=2]        
    }
  }
  return
}

func @conv2_tiled(%I: memref<1x6x5x7xf32>, %K: memref<1x1x7x11xf32>, %O: memref<1x6x5x11xf32>) {
  affine.parallel (%x0, %y) = (0, 0) to (6, 5) step (2, 1) {
    affine.parallel (%x1, %ci, %co) = (%x0, 0, 0) to (%x0 + 2, 7, 11) {
      %0 = pxa.load %I[0, %x1, %y, %ci] : memref<1x6x5x7xf32>
      // CHECK: strides: 0:[^bb0:%arg0=35, ^bb0:%arg1=1, ^bb1:%arg0=70, ^bb1:%arg1=7]
      %1 = pxa.load %K[0, 0, %ci, %co] : memref<1x1x7x11xf32>
      // CHECK: strides: 0:[^bb0:%arg1=11, ^bb0:%arg2=1]
      %2 = mulf %0, %1 : f32
      pxa.reduce addf %2, %O[0, %x1, %y, %co] : memref<1x6x5x11xf32>
      // CHECK: strides: 0:[^bb0:%arg0=55, ^bb0:%arg2=1, ^bb1:%arg0=110, ^bb1:%arg1=11]
    }
  }
  return
}
