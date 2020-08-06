// RUN: pmlc-opt -pxa-vectorize-example %s | FileCheck %s

// CHECK-LABEL: func @pxa_load_0
func @pxa_load_0(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) -> (memref<64x64xf32>) {
  %a = alloc() : memref<64x64xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (64, 64, 66) reduce ("assign") -> (memref<64x64xf32>) {
    %0 = affine.load %arg1[%i, %k] : memref<64x64xf32>
    %1 = affine.load %arg0[%k, %j] : memref<64x64xf32>
    %2 = mulf %0, %1 : f32
    %red = pxa.reduce add %2, %a[%i, %j] :  memref<64x64xf32>
    // CHECK-Lubo: %[[MUL:.*]] = mulf %{{.*}}, %{{.*}} : f32
    // CHECK-Lubo: %{{.*}} = affine.load %[[ARG2:.*]][%[[ARG3:.*]], %[[ARG4:.*]]] : memref<64x64xf32>
    // CHECK-Lubo: affine.store %[[MUL]], %[[ARG2]][%[[ARG3]], %[[ARG4]]] : memref<100x100xf32>
    affine.yield %red : memref<64x64xf32>
  }
  return %r : memref<64x64xf32>
}

