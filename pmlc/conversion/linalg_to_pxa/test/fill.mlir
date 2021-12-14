// RUN: pmlc-opt -convert-linalg-to-pxa -cse %s | FileCheck %s

func @main() -> tensor<16x16xf32> {
  %cst = constant 0.000000e+00 : f32
  %init = linalg.init_tensor [16, 16] : tensor<16x16xf32>
  %fill = linalg.fill(%cst, %init) : f32, tensor<16x16xf32> -> tensor<16x16xf32>
  return %fill : tensor<16x16xf32>
}

// CHECK-LABEL: func @main
//  CHECK-SAME: (%[[arg0:.*]]: memref<16x16xf32>) -> memref<16x16xf32>
//       CHECK:   %[[cst:.*]] = constant 0.000000e+00 : f32
//       CHECK:   %[[out0:.*]] = affine.parallel (%[[arg1:.*]], %[[arg2:.*]]) = (0, 0) to (16, 16) reduce ("assign") -> (memref<16x16xf32>)
//       CHECK:     %[[t0:.*]] = pxa.reduce assign %[[cst]], %[[arg0]][%[[arg1]], %[[arg2]]] : memref<16x16xf32>
//       CHECK:     affine.yield %[[t0]] : memref<16x16xf32>
//       CHECK:   return %[[out0]] : memref<16x16xf32>
