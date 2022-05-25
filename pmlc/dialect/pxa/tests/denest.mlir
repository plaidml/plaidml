// RUN: pmlc-opt -pxa-normalize="denest=true" -canonicalize %s | FileCheck %s

func.func @testDenest(%A: memref<10x10xf32>) -> memref<10x10xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = affine.parallel (%i) = (0) to (10) reduce ("assign") -> memref<10x10xf32> {
    %1 = affine.parallel (%j) = (0) to (10) reduce ("assign") -> memref<10x10xf32> {
      %2 = pxa.reduce assign %cst, %A[%i, %j] : memref<10x10xf32>
      affine.yield %2 : memref<10x10xf32>
    }
    affine.yield %1 : memref<10x10xf32>
  }
  return %0 : memref<10x10xf32>
}

// CHECK: affine.parallel (%[[i:.*]], %[[j:.*]]) = (0, 0) to (10, 10)
// CHECK:   pxa.reduce assign %cst, %{{.*}}[%[[i]], %[[j]]]

