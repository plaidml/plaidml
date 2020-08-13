// RUN: pmlc-opt -pxa-tile-accumulate %s | FileCheck %s

// CHECK-LABEL: func @mixed
func @mixed() {
  // CHECK: alloc()
  // CHECK-NEXT: constant
  %ret_s = alloc() : memref<3xf32>
  %cst_s = constant 0xFF800000 : f32
  // CHECK-NEXT: affine.parallel
  %serial = affine.parallel (%arg0, %arg1) = (0, 0) to (3, 3) reduce ("assign") -> (memref<3xf32>) {
    // CHECK-NEXT: affine.parallel
    // CHECK-NEXT: pxa.reduce assign
    %0 = pxa.reduce assign %cst_s, %ret_s[%arg1] : memref<3xf32>
    // CHECK-NEXT: affine.yield
    affine.yield %0 : memref<3xf32>
  }
  // CHECK: alloc()
  %ret_p = alloc() : memref<3x3xf32>
  // CHECK-NEXT: constant
  %cst_p = constant 0xFF800000 : f32
  // CHECK-NEXT: affine.parallel
  %parallel = affine.parallel (%arg2, %arg3) = (0, 0) to (3, 3) reduce ("assign") -> (memref<3x3xf32>) {
    // CHECK-NEXT: pxa.reduce assign
    %1 = pxa.reduce assign %cst_p, %ret_p[%arg2, %arg3] : memref<3x3xf32>
    // CHECK-NEXT: affine.yield
    affine.yield %1 : memref<3x3xf32>
  }
  return
}