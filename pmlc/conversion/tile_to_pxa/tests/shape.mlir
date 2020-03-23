// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

func @shape(%arg0: tensor<10x20xf32>) -> tensor<2xsi32> {
  %0 = "tile.shape"(%arg0) : (tensor<10x20xf32>) -> tensor<2xsi32>
  return %0 : tensor<2xsi32>
}

// CHECK-LABEL: func @shape
// CHECK-DAG: %[[c0:.*]] = constant 0 : index
// CHECK-DAG: %[[c10:.*]] = constant 10 : i32
// CHECK-DAG: %[[c1:.*]] = constant 1 : index
// CHECK-DAG: %[[c20:.*]] = constant 20 : i32
// CHECK: store %[[c10]], %{{.*}}[%[[c0]]] : memref<2xi32>
// CHECK: store %[[c20]], %{{.*}}[%[[c1]]] : memref<2xi32>
