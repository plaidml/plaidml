// RUN: pmlc-opt -convert-linalg-to-pxa -cse %s | FileCheck %s

func @main() -> tensor<16x16xf32> {
  // CHECK: memref.alloc
  %init = linalg.init_tensor [16, 16] : tensor<16x16xf32>
  return %init : tensor<16x16xf32>
}
