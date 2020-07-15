// RUN: pmlc-opt -pxa-localize -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @simple
func @simple(%arg0: memref<3x4xf16>) -> memref<3x4xf16> {
  %cst = constant 0.0 : f16
  // CHECK: alloc()
  %0 = alloc() : memref<3x4xf16>
  %1 = alloc() : memref<3x4xf16>
  // CHECK-NEXT: affine.parallel
  %2 = affine.parallel (%i) = (0) to (3) reduce ("assign") -> (memref<3x4xf16>) {
    // CHECK-NEXT: alloc
    // CHECK-NEXT: affine.parallel
    %3 = affine.parallel (%j) = (0) to (4) reduce ("assign") -> (memref<3x4xf16>) {
      %5 = pxa.reduce assign %cst, %0[%i, %j] : memref<3x4xf16>
      %6 = pxa.reduce add %cst, %5[%i, %j] : memref<3x4xf16>
      affine.yield %6 : memref<3x4xf16>
    }
    %4 = affine.parallel (%j) = (0) to (4) reduce ("assign") -> (memref<3x4xf16>) {
      %5 = affine.load %3[%i, %j] : memref<3x4xf16>
      %6 = pxa.reduce assign %5, %1[%i, %j] : memref<3x4xf16>
      affine.yield %6 : memref<3x4xf16>
    }
    affine.yield %4 : memref<3x4xf16>
  }
  return %2 : memref<3x4xf16>
}
