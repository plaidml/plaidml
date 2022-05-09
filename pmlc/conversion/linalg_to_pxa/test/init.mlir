// RUN: pmlc-opt -convert-linalg-to-pxa -cse %s | FileCheck %s

// CHECK: func @main(%arg0: memref<16x16xf32>) -> memref<16x16xf32> {
func @main() -> tensor<16x16xf32> {
  %init = linalg.init_tensor [16, 16] : tensor<16x16xf32>
  return %init : tensor<16x16xf32>
}
