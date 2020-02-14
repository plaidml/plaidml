// RUN: pmlc-opt -canonicalize -test-stride-info  %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0, 0, 0)>
#map2 = affine_map<() -> (100, 100, 100)>

module {
  func @simple(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>, %arg2: memref<100x100xf32>) {
    affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) {
      %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
      // CHECK: stride begin
      // CHECK: offset = 0
      // CHECK-DAG: ba0 = 100
      // CHECK-DAG: ba2 = 1
      // CHECK: stride end
      %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
      // CHECK: stride begin
      // CHECK: offset = 0
      // CHECK-DAG: ba2 = 100
      // CHECK-DAG: ba1 = 1
      // CHECK: stride end
      %2 = mulf %0, %1 : f32
      pxa.reduce add %2, %arg2[%i, %j] : memref<100x100xf32>
    }
    return
  }
  func @symbolic_fail(%arg0: memref<100x?xf32>) {
    %d1 = dim %arg0, 0 : memref<100x?xf32>
    affine.parallel (%i, %j) = (0, 0) to (100, symbol(%d1)) {
      %0 = affine.load %arg0[%i, %j] : memref<100x?xf32>
      // CHECK: stride begin
      // CHECK: none
      // CHECK: stride end
    }
    return
  }
  func @for_diagonal(%arg0: memref<100x100xf32>) {
    affine.for %i = 0 to 10 {
      %0 = affine.load %arg0[%i, %i] : memref<100x100xf32>
      // CHECK: stride begin
      // CHECK: offset = 0
      // CHECK: ba0 = 101 
      // CHECK: stride end
    }
    return
  }
  func @for_step(%arg0: memref<100x100xf32>) {
    affine.for %i = 0 to 10 step 2 {
      %0 = affine.load %arg0[%i, %i] : memref<100x100xf32>
      // CHECK: stride begin
      // CHECK: offset = 0
      // CHECK: ba0 = 202 
      // CHECK: stride end
    }
    return
  }
  func @parallel_step(%arg0: memref<100x100xf32>) {
    affine.parallel (%i, %j) = (0, 0) to (10, 10) step (2, 5) {
      %0 = affine.load %arg0[%i, %j] : memref<100x100xf32>
      // CHECK: stride begin
      // CHECK: offset = 0
      // CHECK-DAG: ba0 = 200
      // CHECK-DAG: ba1 = 5 
      // CHECK: stride end
    }
    return
  }
  func @parallel_tile(%arg0: memref<100x100xf32>) {
    affine.parallel (%i, %j) = (0, 0) to (100, 100) step (10, 10) {
      affine.parallel (%i2, %j2) = (%i, %j)  to (%i + 10, %j + 10) step (1, 1) {
        %0 = affine.load %arg0[%i2, %j2] : memref<100x100xf32>
        // CHECK: stride begin
        // CHECK: offset = 0
        // CHECK-DAG: ba0 = 1000
        // CHECK-DAG: ba0 = 100
        // CHECK-DAG: ba1 = 10
        // CHECK-DAG: ba1 = 1
        // CHECK: stride end
      }
    }
    return
  }
  func @affine_apply(%arg0: memref<100x100xf32>) {
    affine.for %i = 0 to 10 {
      %0 = affine.apply affine_map<(d1) -> (5 * d1)>(%i)
      %1 = affine.load %arg0[%0, %i] : memref<100x100xf32>
      // CHECK: stride begin
      // CHECK: offset = 0
      // CHECK: ba0 = 501
      // CHECK: stride end
    }
    return
  }
  func @affine_apply_add(%arg0: memref<100x100xf32>) {
    affine.for %i = 0 to 10 {
      %0 = affine.apply affine_map<(d1) -> (d1 + 10)>(%i)
      %1 = affine.load %arg0[%0, %i] : memref<100x100xf32>
      // CHECK: stride begin
      // CHECK: offset = 10 
      // CHECK: ba0 = 101
      // CHECK: stride end
    }
    return
  }
}
