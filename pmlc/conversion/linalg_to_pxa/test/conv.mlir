// RUN: pmlc-opt -convert-linalg-to-pxa %s | FileCheck %s

module  {
  func @test_conv(%arg0: memref<128x128x3xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<3x3x3xf32, offset: ?, strides: [?, ?, 1]>, %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
    linalg.conv(%arg0, %arg1, %arg2) {strides = [2]}: memref<128x128x3xf32, offset: ?, strides: [?, ?, 1]>, memref<3x3x3xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
    return
  }
}

// CHECK: #map = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>
// CHECK-LABEL: func @test_conv
// CHECK-SAME: (%[[arg0:.*]]: memref<128x128x3xf32, #map>, %[[arg1:.*]]: memref<3x3x3xf32, #map>, %[[arg2:.*]]: memref<?x?x?xf32, #map>)
// CHECK: affine.parallel (%[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]], %[[arg6:.*]], %[[arg7:.*]]) = (0, 0, 0, 0, 0) to (3, -1, 3, 128, 128) reduce ("addf")
// CHECK:   %[[t0:.*]] = pxa.load %[[arg0]][%[[arg7]], %[[arg6]], %[[arg5]]] : memref<128x128x3xf32, #map>
// CHECK:   %[[t1:.*]] = pxa.load %[[arg1]][%[[arg3]], %[[arg4]] * 2 + %[[arg7]], %[[arg6]]] : memref<3x3x3xf32, #map>
// CHECK:   %[[t2:.*]] = mulf %[[t0]], %[[t1]] : f32
// CHECK:   %[[t3:.*]] = pxa.reduce addf %[[t2]], %[[arg2]][%[[arg3]], %[[arg4]], %[[arg5]]] : memref<?x?x?xf32, #map>
// CHECK:   affine.yield %[[t3]] : memref<?x?x?xf32, #map>
// CHECK: return

