// RUN: pmlc-opt -convert-tile-to-linalg %s | FileCheck %s

func.func @shape(%arg0: tensor<10x20xf32>) -> tensor<2xsi32> {
  %0 = "tile.shape"(%arg0) : (tensor<10x20xf32>) -> tensor<2xsi32>
  return %0 : tensor<2xsi32>
}

// CHECK-LABEL: func.func @shape
// CHECK: %[[cst:.*]] = arith.constant dense<[10, 20]> : tensor<2xi32>
// CHECK: return %[[cst]] : tensor<2xi32>
