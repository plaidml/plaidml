// RUN: pmlc-opt -stdx-check-bounds %s | FileCheck %s

module {
  func @simpleStore(%A: memref<20x10xf32>, %i: index, %j: index, %f: f32) -> () {
    store %f, %A[%i, %j] : memref<20x10xf32>
    return
  }
  // CHECK: %{{.*}} = constant 20 : i64
  // CHECK-NEXT call @plaidml_rt_bounds_check(%arg1, %{{.*}}) : (index, i64) -> ()
  // CHECK-NEXT %{{.*}} = constant 10 : i64
  // CHECK-NEXT call @plaidml_rt_bounds_check(%arg2, %{{.*}}) : (index, i64) -> ()
  // CHECK-NEXT store %arg3, %arg0[%arg1, %arg2] : memref<20x10xf32>
}