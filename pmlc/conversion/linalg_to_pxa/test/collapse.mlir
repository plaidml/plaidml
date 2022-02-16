// RUN: pmlc-opt -convert-linalg-to-pxa -cse %s | FileCheck %s

func @test_collapse_reshape(%arg0: tensor<3x4x5xf32>) -> tensor<12x5xf32> {
  %0 = linalg.tensor_collapse_shape %arg0 [[0, 1], [2]] : tensor<3x4x5xf32> into tensor<12x5xf32>
  return %0 : tensor<12x5xf32>
}

// CHECK-LABEL: func @test_collapse_reshape
//  CHECK-SAME: (%[[arg0:.*]]: memref<3x4x5xf32>, %[[arg1:.*]]: memref<12x5xf32>) -> memref<12x5xf32>
//       CHECK:   %[[out0:.*]] = affine.parallel (%[[arg2:.*]], %[[arg3:.*]], %[[arg4:.*]]) = (0, 0, 0) to (3, 4, 5) reduce ("assign") -> (memref<12x5xf32>)
//       CHECK:     %[[t0:.*]] = pxa.load %[[arg0]][%[[arg2]], %[[arg3]], %[[arg4]]] : memref<3x4x5xf32>
//       CHECK:     %[[t1:.*]] = pxa.reduce assign %[[t0]], %[[arg1]][%[[arg2]] * 4 + %[[arg3]], %[[arg4]]] : memref<12x5xf32>
//       CHECK:     affine.yield %[[t1]] : memref<12x5xf32>
//       CHECK:   return %[[out0]] : memref<12x5xf32>

func @zero_dim(%arg0: tensor<1x1x1xi32>) -> tensor<i32> {
  %0 = linalg.tensor_collapse_shape %arg0 [] : tensor<1x1x1xi32> into tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func @zero_dim
//  CHECK-SAME: (%[[arg0:.*]]: memref<1x1x1xi32>, %[[arg1:.*]]: memref<i32>) -> memref<i32>
//       CHECK:   %[[out:.*]] = affine.parallel (%[[arg2:.*]], %[[arg3:.*]], %[[arg4:.*]]) = (0, 0, 0) to (1, 1, 1) reduce ("assign") -> (memref<i32>)
//       CHECK:     %[[t0:.*]] = pxa.load %[[arg0]][%[[arg2]], %[[arg3]], %[[arg4]]] : memref<1x1x1xi32>
//       CHECK:     %[[t1:.*]] = pxa.reduce assign %[[t0]], %[[arg1]][] : memref<i32>
//       CHECK:     affine.yield %[[t1]] : memref<i32>
//       CHECK:   return %[[out]] : memref<i32>
