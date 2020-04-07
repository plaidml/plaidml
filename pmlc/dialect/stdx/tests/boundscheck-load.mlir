// RUN: pmlc-opt -stdx-check-bounds %s | FileCheck %s

module {
  func @simpleLoad(%A: memref<20x10xf32>, %i: index, %j: index) -> (f32) {
    %0 = load %A[%i, %j] : memref<20x10xf32>
    return %0: f32
  }
  // CHECK: %{{.*}} = constant 20 : i32
  // CHECK-NEXT call @plaidml_rt_bounds_check(%arg1, %{{.*}}) : (index, i32) -> ()
  // CHECK-NEXT %{{.*}} = constant 10 : i32
  // CHECK-NEXT call @plaidml_rt_bounds_check(%arg2, %{{.*}}) : (index, i32) -> ()
  // CHECK-NEXT %{{.*}} = load %arg0[%arg1, %arg2] : memref<20x10xf32>
}