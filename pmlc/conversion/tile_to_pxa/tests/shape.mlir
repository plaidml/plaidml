// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

func @shape(%arg0: tensor<10x20xf32>) -> tensor<2xsi32> {
  %0 = "tile.shape"(%arg0) : (tensor<10x20xf32>) -> tensor<2xsi32>
  return %0 : tensor<2xsi32>
}

// CHECK-LABEL: func @shape
// CHECK-DAG: %[[c10:.*]] = arith.constant 10 : i32
// CHECK-DAG: %[[c20:.*]] = arith.constant 20 : i32
// CHECK: %[[r0:.*]] = pxa.reduce assign %[[c10]], %{{.*}}[0] : memref<2xi32>
// CHECK: %[[r1:.*]] = pxa.reduce assign %[[c20]], %[[r0]][1] : memref<2xi32>
// CHECK: return %[[r1]] : memref<2xi32>
