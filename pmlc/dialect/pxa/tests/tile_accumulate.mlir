// RUN: pmlc-opt -pxa-tile-accumulate %s | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (10)>
#map3 = affine_map<() -> (0, 0)>
#map4 = affine_map<() -> (10, 10)>

#set0 = affine_set<(d0, d1) : (d0 - d1 >= 0)>

// CHECK-LABEL: func @cumsum
// CHECK-SAME: (%[[arg0:.*]]: memref<10xf32> {tile.name = "I"}) -> memref<10xf32>
func @cumsum(%arg0: memref<10xf32> {tile.name = "I"}) -> memref<10xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = alloc() : memref<10xf32>
  %1 = affine.parallel (%arg1) = (0) to (10) reduce ("assign") -> (memref<10xf32>) {
    %3 = pxa.reduce assign %cst, %0[%arg1] : memref<10xf32>
    affine.yield %3 : memref<10xf32>
  }
  %2 = affine.parallel (%arg1, %arg2) = (0, 0) to (10, 10) reduce ("assign") -> (memref<10xf32>) {
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

// CHECK: %[[cst:.*]] = constant 0.000000e+00 : f32
// CHECK: %[[A0:.*]] = alloc() : memref<10xf32>
// CHECK: %[[P1:.*]] = affine.parallel (%[[arg1:.*]]) = (0) to (10) reduce ("assign") -> (memref<10xf32>) {
// CHECK:   %[[P3:.*]] = affine.parallel (%[[arg2:.*]]) = (0) to (1) reduce ("assign") -> (memref<10xf32>) {
// CHECK:     %[[P4:.*]] = pxa.reduce assign %[[cst]], %[[A0]][%[[arg1]]] : memref<10xf32>
// CHECK:     affine.yield %[[P4]] : memref<10xf32>
// CHECK:   affine.yield %[[P3]] : memref<10xf32>
// CHECK: %[[P2:.*]] = affine.parallel (%[[arg1:.*]], %[[arg2:.*]]) = (0, 0) to (10, 10) step (1, 10) reduce ("assign") -> (memref<10xf32>)
// CHECK:   %[[P3:.*]] = affine.parallel (%[[arg3:.*]], %[[arg4:.*]]) = (%[[arg1]], %[[arg2]]) to (%[[arg1]] + 1, %[[arg2]] + 10) reduce ("assign") -> (memref<10xf32>)
// CHECK:     %[[P4:.*]] = affine.if #set0(%[[arg3]], %[[arg4]]) -> memref<10xf32>
// CHECK:       %[[L5:.*]] = affine.load %[[arg0]][%[[arg4]]] : memref<10xf32>
// CHECK:       %[[R6:.*]] = pxa.reduce addf %[[L5]], %[[P1]][%[[arg3]]] : memref<10xf32>
// CHECK:       affine.yield %[[R6]] : memref<10xf32>
// CHECK:       affine.yield %[[P1]] : memref<10xf32>
// CHECK:     affine.yield %[[P4]] : memref<10xf32>
// CHECK:   affine.yield %[[P3]] : memref<10xf32>
// CHECK: return %[[P2]] : memref<10xf32>

// CHECK-LABEL: func @mixed
func @mixed() {
  %ret_s = alloc() : memref<3xf32>
  %cst_s = constant 0xFF800000 : f32
  %serial = affine.parallel (%arg0, %arg1) = (0, 0) to (3, 3) reduce ("assign") -> (memref<3xf32>) {
    %0 = pxa.reduce assign %cst_s, %ret_s[%arg1] : memref<3xf32>
    affine.yield %0 : memref<3xf32>
  }
  %ret_p = alloc() : memref<3x3xf32>
  %cst_p = constant 0xFF800000 : f32
  %parallel = affine.parallel (%arg2, %arg3) = (0, 0) to (3, 3) reduce ("assign") -> (memref<3x3xf32>) {
    %1 = pxa.reduce assign %cst_p, %ret_p[%arg2, %arg3] : memref<3x3xf32>
    affine.yield %1 : memref<3x3xf32>
  }
  return
}

// CHECK: alloc() : memref<3xf32>
// CHECK: constant 0xFF800000 : f32
// CHECK: affine.parallel (%[[arg0:.*]], %[[arg1:.*]]) = (0, 0) to (3, 3) step (3, 1) reduce ("assign") -> (memref<3xf32>)
// CHECK:   affine.parallel (%[[arg2:.*]], %[[arg3:.*]]) = (%[[arg0]], %[[arg1]]) to (%[[arg0]] + 3, %[[arg1]] + 1) reduce ("assign") -> (memref<3xf32>)
// CHECK:     pxa.reduce assign %{{.*}}, %{{.*}}[%[[arg3]]] : memref<3xf32>
// CHECK:       affine.yield
// CHECK:     affine.yield
// CHECK: alloc() : memref<3x3xf32>
// CHECK: constant 0xFF800000 : f32
// CHECK: affine.parallel (%[[arg0:.*]], %[[arg1:.*]]) = (0, 0) to (3, 3) reduce ("assign") -> (memref<3x3xf32>)
// CHECK:   pxa.reduce assign %{{.*}}, %{{.*}}[%[[arg0]], %[[arg1]]] : memref<3x3xf32>
// CHECK:   affine.yield
// CHECK: return

// CHECK-LABEL: func @const_add
// CHECK-SAME: (%[[arg0:.*]]: memref<4xi32> {tile.name = "B"}, %[[arg1:.*]]: memref<4xi32> {tile.name = "A"}) -> memref<4xi32>
func @const_add(%arg0: memref<4xi32> {tile.name = "B"}, %arg1: memref<4xi32> {tile.name = "A"}) -> memref<4xi32> {
  // CHECK: alloc() : memref<4xi32>
  %0 = alloc() : memref<4xi32>
  // CHECK-NEXT: affine.parallel (%[[arg2:.*]]) = (0) to (4) reduce ("assign") -> (memref<4xi32>)
  // CHECK-NEXT: affine.parallel (%[[arg3:.*]]) = (0) to (1) reduce ("assign") -> (memref<4xi32>)
  %1 = affine.parallel (%arg2) = (0) to (4) reduce ("assign") -> (memref<4xi32>) {
    %2 = pxa.load %arg1[%arg2] : memref<4xi32>
    %3 = pxa.load %arg0[%arg2] : memref<4xi32>
    %4 = addi %2, %3 : i32
    %5 = pxa.reduce assign %4, %0[%arg2] : memref<4xi32>
    affine.yield %5 : memref<4xi32>
  }
  return %1 : memref<4xi32>
}