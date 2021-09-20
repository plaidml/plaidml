// RUN: pmlc-opt -convert-linalg-to-pxa %s | FileCheck %s

func @main(%arg0: tensor<3xf32> {stdx.const}) -> tensor<3xf32> {
  return %arg0 : tensor<3xf32>
}

// CHECK-LABEL: func @main
// CHECK-SAME: (%[[arg0:.*]]: memref<3xf32> {stdx.const}, {{.*}}: memref<3xf32>) -> memref<3xf32>
// CHECK:   return %[[arg0]] : memref<3xf32>
