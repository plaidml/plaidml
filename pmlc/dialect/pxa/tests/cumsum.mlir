// RUN: pmlc-opt -pxa-tile-accumulate %s | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (10)>
#map3 = affine_map<() -> (0, 0)>
#map4 = affine_map<() -> (10, 10)>

#set0 = affine_set<(d0, d1) : (d0 - d1 >= 0)>

// CHECK-LABEL: func @cumsum
func @cumsum(%arg0: memref<10xf32> {tile.name = "I"}) -> memref<10xf32> {
%cst = constant 0.000000e+00 : f32
%0 = alloc() : memref<10xf32>
// CHECK: affine.parallel
%1 = affine.parallel (%arg1) = (0) to (10) reduce ("assign") -> (memref<10xf32>) {
  // CHECK-NEXT: affine.parallel
  %3 = pxa.reduce assign %cst, %0[%arg1] : memref<10xf32>
  affine.yield %3 : memref<10xf32>
}
// CHECK: affine.parallel
%2 = affine.parallel (%arg1, %arg2) = (0, 0) to (10, 10) reduce ("assign") -> (memref<10xf32>) {
  // CHECK-NEXT: affine.parallel
  %3 = affine.if #set0(%arg1, %arg2) -> memref<10xf32> {
    %4 = affine.load %arg0[%arg2] : memref<10xf32>
    %5 = pxa.reduce addf %4, %1[%arg1] : memref<10xf32>
    affine.yield %5 : memref<10xf32>
  } else {
    affine.yield %1 : memref<10xf32>
  }
  affine.yield %3 : memref<10xf32>
}
return %2 : memref<10xf32>
}
