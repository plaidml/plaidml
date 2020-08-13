// RUN: pmlc-opt -pxa-tile-accumulate %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (0, d0, d1, d2)>
#map1 = affine_map<() -> (0, 0, 0)>
#map2 = affine_map<() -> (224, 224, 32)>
#map3 = affine_map<(d0, d1, d2, d3) -> (0, d0 + d2 - 1, d1 + d3 - 1, 0)>
#map4 = affine_map<(d0, d1, d2) -> (d1, d2, 0, d0)>
#map5 = affine_map<() -> (0, 0, 0, 0, 0)>
#map6 = affine_map<() -> (224, 224, 32, 3, 3)>

#set0 = affine_set<(d0, d1, d2, d3) : (d0 + d2 - 1 >= 0, -d0 - d2 + 224 >= 0, d1 + d3 - 1 >= 0, -d1 - d3 + 224 >= 0)>

// CHECK-LABEL: func @convolution
func @convolution(%arg0: memref<3x3x1x32xf32>, %arg1: memref<1x224x224x3xf32>) -> memref<1x224x224x32xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = alloc() : memref<1x224x224x32xf32>
  %1 = affine.parallel (%arg2, %arg3, %arg4) = (0, 0, 0) to (224, 224, 32) reduce ("assign") -> (memref<1x224x224x32xf32>) {
    %3 = pxa.reduce assign %cst, %0[0, %arg2, %arg3, %arg4] : memref<1x224x224x32xf32>
    affine.yield %3 : memref<1x224x224x32xf32>
  }
  %2 = affine.parallel (%arg2, %arg3, %arg4, %arg5, %arg6) = (0, 0, 0, 0, 0) to (224, 224, 32, 3, 3) reduce ("assign") -> (memref<1x224x224x32xf32>) {
    %3 = affine.if #set0(%arg2, %arg3, %arg5, %arg6) -> memref<1x224x224x32xf32> {
      %4 = affine.load %arg1[0, %arg2 + %arg5 - 1, %arg3 + %arg6 - 1, 0] : memref<1x224x224x3xf32>
      %5 = affine.load %arg0[%arg5, %arg6, 0, %arg4] : memref<3x3x1x32xf32>
      %6 = mulf %4, %5 : f32
      %7 = pxa.reduce addf %6, %1[0, %arg2, %arg3, %arg4] : memref<1x224x224x32xf32>
      affine.yield %7 : memref<1x224x224x32xf32>
    } else {
      affine.yield %1 : memref<1x224x224x32xf32>
    }
    affine.yield %3 : memref<1x224x224x32xf32>
  }
  return %2 : memref<1x224x224x32xf32>
  // CHECK: constant 0.000000e+00 : f32
  // CHECK: alloc() : memref<1x224x224x32xf32>
  // CHECK: affine.parallel
  // CHECK:   pxa.reduce assign
  // CHECK:   affine.yield
  // CHECK: affine.parallel
  // CHECK: affine.if
  // CHECK:   affine.load
  // CHECK:   affine.load
  // CHECK:   mulf
  // CHECK:   pxa.reduce
  // CHECK:   affine.yield
  // CHECK:   affine.yield
  // CHECK: affine.yield
}