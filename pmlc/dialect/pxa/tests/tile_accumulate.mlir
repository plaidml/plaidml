// RUN: pmlc-opt -pxa-tile-accumulate %s | FileCheck %s

// CHECK-LABEL: func @cumsum
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (10)>
#map3 = affine_map<() -> (0, 0)>
#map4 = affine_map<() -> (10, 10)>

#set0 = affine_set<(d0, d1) : (d0 - d1 >= 0)>

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
