// RUN: pmlc-opt -tile-legalize-to-pxa -canonicalize -cse %s | FileCheck %s

func @shape(%arg0: tensor<10x20x!eltwise.fp32>) -> tensor<2x!eltwise.i32> {
  %0 = "tile.shape"(%arg0) : (tensor<10x20x!eltwise.fp32>) -> tensor<2x!eltwise.i32>
  return %0 : tensor<2x!eltwise.i32>
}

// CHECK-LABEL: func @shape
// CHECK: %c0 = constant 0 : index
// CHECK: %c10 = constant 10 : index
// CHECK: %c1 = constant 1 : index
// CHECK: %c20 = constant 20 : index
// CHECK: %0 = index_cast %c10 : index to i32
// CHECK: store %0, %arg1[%c0] : memref<2xi32>
// CHECK: %1 = index_cast %c20 : index to i32
// CHECK: store %1, %arg1[%c1] : memref<2xi32>
