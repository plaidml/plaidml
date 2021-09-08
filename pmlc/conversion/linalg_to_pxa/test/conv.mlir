// RUN: pmlc-opt -convert-linalg-to-pxa -cse %s | FileCheck %s

module  {
  func @test_conv(%arg0: memref<128x128x3xf32>, %arg1: memref<3x3x3xf32>, %arg2: memref<126x126x3xf32>) {
    linalg.conv(%arg0, %arg1, %arg2) {strides = [2]}: memref<128x128x3xf32>, memref<3x3x3xf32>, memref<126x126x3xf32>
    return
  }
}

// CHECK-LABEL: func @test_conv
// CHECK-SAME: (%[[arg0:.*]]: memref<128x128x3xf32>, %[[arg1:.*]]: memref<3x3x3xf32>, %[[arg2:.*]]: memref<126x126x3xf32>)
// CHECK: affine.parallel (%[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]], %[[arg6:.*]], %[[arg7:.*]]) = (0, 0, 0, 0, 0) to (3, 126, 3, 128, 128) reduce ("addf")
// CHECK:   %[[t0:.*]] = pxa.load %[[arg0]][%[[arg7]], %[[arg6]], %[[arg5]]] : memref<128x128x3xf32>
// CHECK:   %[[t1:.*]] = pxa.load %[[arg1]][%[[arg3]], %[[arg4]] * 2 + %[[arg7]], %[[arg6]]] : memref<3x3x3xf32>
// CHECK:   %[[t2:.*]] = mulf %[[t0]], %[[t1]] : f32
// CHECK:   %[[t3:.*]] = pxa.reduce addf %[[t2]], %[[arg2]][%[[arg3]], %[[arg4]], %[[arg5]]] : memref<126x126x3xf32>
// CHECK:   affine.yield %[[t3]] : memref<126x126x3xf32>
// CHECK: return

