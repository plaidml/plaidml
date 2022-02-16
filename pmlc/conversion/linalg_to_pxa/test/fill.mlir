// RUN: pmlc-opt -convert-linalg-to-pxa -cse %s | FileCheck %s

func @main() -> tensor<16x16xf32> {
  %cst = constant 0.000000e+00 : f32
  %init = linalg.init_tensor [16, 16] : tensor<16x16xf32>
  %fill = linalg.fill(%cst, %init) : f32, tensor<16x16xf32> -> tensor<16x16xf32>
  return %fill : tensor<16x16xf32>
}

//CHECK: module
// CHECK:  memref.global "private" constant @cst_scalar_memref_0 : memref<f32> = dense<0.000000e+00>
// CHECK-LABEL: func @main
//  CHECK-SAME: (%[[arg0:.*]]: memref<16x16xf32>) -> memref<16x16xf32>
//       CHECK:  %[[cst0:.*]] = memref.get_global @cst_scalar_memref_0 : memref<f32>
//       CHECK:  %[[t0:.*]]  = pxa.load %[[cst0]][] : memref<f32>
//       CHECK:   %[[out0:.*]] = affine.parallel (%[[arg1:.*]], %[[arg2:.*]]) = (0, 0) to (16, 16) reduce ("assign") -> (memref<16x16xf32>)
//       CHECK:     %[[t1:.*]] = pxa.reduce assign %[[t0]], %[[arg0]][%[[arg1]], %[[arg2]]] : memref<16x16xf32>
//       CHECK:     affine.yield %[[t1]] : memref<16x16xf32>
//       CHECK:   return %[[out0]] : memref<16x16xf32>


 

