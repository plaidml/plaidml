// RUN: pmlc-opt -convert-linalg-to-pxa -cse %s | FileCheck %s

func @pooling_sum(%arg0: memref<8x8xf32>, %arg1: memref<2x2xi32>, %arg2: memref<7x7xf32>) -> memref<7x7xf32>{
  linalg.pooling_sum(%arg0, %arg1, %arg2) : memref<8x8xf32>, memref<2x2xi32>, memref<7x7xf32>
  return %arg2 : memref<7x7xf32>
}

// CHECK-LABEL: func @pooling_sum
// CHECK-SAME: (%[[arg0:.*]]: memref<8x8xf32>, %[[arg1:.*]]: memref<2x2xi32>, %[[arg2:.*]]: memref<7x7xf32>) -> memref<7x7xf32>
// CHECK: %[[out0:.*]] = affine.parallel (%[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]], %[[arg6:.*]]) = (0, 0, 0, 0) to (7, 7, 2, 2) reduce ("addf") -> (memref<7x7xf32>)
// CHECK:   %[[t0:.*]] = pxa.load %[[arg0]][%[[arg3]] + %[[arg5]], %[[arg4]] + %[[arg6]]] : memref<8x8xf32>
// CHECK:   %[[t2:.*]] = pxa.reduce addf %[[t0]], %[[arg2]][%[[arg3]], %[[arg4]]] : memref<7x7xf32>
// CHECK:   affine.yield %[[t2]] : memref<7x7xf32>
// CHECK: return %[[out0]] : memref<7x7xf32>

func @pooling_max(%arg0: memref<8x8xf32>, %arg1: memref<2x2xi32>, %arg2: memref<7x7xf32>) -> memref<7x7xf32>{
  linalg.pooling_max(%arg0, %arg1, %arg2) : memref<8x8xf32>, memref<2x2xi32>, memref<7x7xf32>
  return %arg2 : memref<7x7xf32>
}

// CHECK-LABEL: func @pooling_max
// CHECK-SAME: (%[[arg0:.*]]: memref<8x8xf32>, %[[arg1:.*]]: memref<2x2xi32>, %[[arg2:.*]]: memref<7x7xf32>) -> memref<7x7xf32>
// CHECK: %[[out0:.*]] = affine.parallel (%[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]], %[[arg6:.*]]) = (0, 0, 0, 0) to (7, 7, 2, 2) reduce ("maxf") -> (memref<7x7xf32>)
// CHECK:   %[[t0:.*]] = pxa.load %[[arg0]][%[[arg3]] + %[[arg5]], %[[arg4]] + %[[arg6]]] : memref<8x8xf32>
// CHECK:   %[[t2:.*]] = pxa.reduce maxf %[[t0]], %[[arg2]][%[[arg3]], %[[arg4]]] : memref<7x7xf32>
// CHECK:   affine.yield %[[t2]] : memref<7x7xf32>
// CHECK: return %[[out0]] : memref<7x7xf32>

func @pooling_min(%arg0: memref<8x8xf32>, %arg1: memref<2x2xi32>, %arg2: memref<7x7xf32>) -> memref<7x7xf32>{
  linalg.pooling_min(%arg0, %arg1, %arg2) : memref<8x8xf32>, memref<2x2xi32>, memref<7x7xf32>
  return %arg2 : memref<7x7xf32>
}

// CHECK-LABEL: func @pooling_min
// CHECK-SAME: (%[[arg0:.*]]: memref<8x8xf32>, %[[arg1:.*]]: memref<2x2xi32>, %[[arg2:.*]]: memref<7x7xf32>) -> memref<7x7xf32>
// CHECK: %[[out0:.*]] = affine.parallel (%[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]], %[[arg6:.*]]) = (0, 0, 0, 0) to (7, 7, 2, 2) reduce ("minf") -> (memref<7x7xf32>)
// CHECK:   %[[t0:.*]] = pxa.load %[[arg0]][%[[arg3]] + %[[arg5]], %[[arg4]] + %[[arg6]]] : memref<8x8xf32>
// CHECK:   %[[t2:.*]] = pxa.reduce minf %[[t0]], %[[arg2]][%[[arg3]], %[[arg4]]] : memref<7x7xf32>
// CHECK:   affine.yield %[[t2]] : memref<7x7xf32>
// CHECK: return %[[out0]] : memref<7x7xf32>
