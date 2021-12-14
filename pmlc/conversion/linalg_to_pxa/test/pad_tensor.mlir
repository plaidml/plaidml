// RUN: pmlc-opt -convert-linalg-to-pxa -cse %s | FileCheck %s

func @test_pad(%arg0: tensor<4x8x8x16xf32>) -> tensor<8x10x14x20xf32> {
  %pad_value = constant 0.000000e+00 : f32
  %0 = linalg.pad_tensor %arg0 low[2, 2, 4, 4] high[2, 0, 2, 0] {
    ^bb0(%arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index):
      linalg.yield %pad_value : f32
  } : tensor<4x8x8x16xf32> to tensor<8x10x14x20xf32>
  return %0 : tensor<8x10x14x20xf32>
}

// CHECK-LABEL: func @test_pad
//  CHECK-SAME: (%[[arg0:.*]]: memref<4x8x8x16xf32>, %[[arg1:.*]]: memref<8x10x14x20xf32>) -> memref<8x10x14x20xf32>
//       CHECK:   %[[cst:.*]] = constant 0.000000e+00 : f32
//       CHECK:   %[[init:.*]] = memref.alloc()
//       CHECK:   %[[out1:.*]] = affine.parallel (%[[arg2:.*]], %[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]]) = (0, 0, 0, 0) to (8, 10, 14, 20)
//       CHECK:     %[[t0:.*]] = pxa.reduce assign %[[cst]], %[[init]][%[[arg2]], %[[arg3]], %[[arg4]], %[[arg5]]] : memref<8x10x14x20xf32>
//       CHECK:     affine.yield %[[t0]] : memref<8x10x14x20xf32>
//       CHECK:   %[[out2:.*]] = affine.parallel (%[[arg2:.*]], %[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]]) = (0, 0, 0, 0) to (8, 10, 14, 20)
//       CHECK:     %[[t1:.*]] = pxa.load %[[out1]][%arg2, %arg3, %arg4, %arg5] : memref<8x10x14x20xf32>
//       CHECK:     %[[t2:.*]] = pxa.reduce assign %[[t1]], %[[arg1]][%[[arg2]], %[[arg3]], %[[arg4]], %[[arg5]]] : memref<8x10x14x20xf32>
//       CHECK:     affine.yield %[[t2]] : memref<8x10x14x20xf32>
//       CHECK:   %[[out3:.*]] = affine.parallel (%[[arg2:.*]], %[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]]) = (0, 0, 0, 0) to (4, 8, 8, 16)
//       CHECK:     %[[t3:.*]] = pxa.load %[[arg0]][%[[arg2]], %[[arg3]], %[[arg4]], %[[arg5]]] : memref<4x8x8x16xf32>
//       CHECK:     %[[t4:.*]] = pxa.reduce assign %[[t3]], %[[out2]][%[[arg2]] + 2, %[[arg3]] + 2, %[[arg4]] + 4, %[[arg5]] + 4] : memref<8x10x14x20xf32>
//       CHECK:     affine.yield %[[t4]] : memref<8x10x14x20xf32>
//       CHECK:   return %[[out3]] : memref<8x10x14x20xf32>
