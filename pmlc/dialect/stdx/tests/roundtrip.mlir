// RUN: pmlc-opt %s | pmlc-opt | FileCheck %s

// CHECK-LABEL: func @dot
func @dot(%A: memref<16x10xf32>, %B: memref<16x10xf32>, %O: memref<16x10xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %c16 = constant 16 : index
  loop.parallel (%i, %j) = (%c0, %c0) to (%c16, %c10) step (%c1, %c1) {
    %0 = load %A[%i, %j] : memref<16x10xf32>
    %1 = load %B[%i, %j] : memref<16x10xf32>
    %2 = mulf %0, %1 : f32
    // CHECK: stdx.atomic_rmw %{{.*}} = %{{.*}}[%{{.*}}, %{{.*}}] : memref<16x10xf32>
    stdx.atomic_rmw %val = %O[%i, %j] : memref<16x10xf32> {
      %3 = addf %val, %2 : f32
      // CHECK: stdx.atomic_rmw.yield %{{.*}} : f32
      stdx.atomic_rmw.yield %3 : f32
    }
  }
  return
}
