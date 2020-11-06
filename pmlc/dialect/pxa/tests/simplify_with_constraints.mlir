// RUN: pmlc-opt -pxa-simplify-with-constraints %s | FileCheck %s

// CHECK-LABEL: func @mod
//       CHECK:   affine.parallel (%[[I:[a-zA-Z0-9]*]], %[[J:[a-zA-Z0-9]*]])
//       CHECK:     pxa.load {{.*}}[%[[I]], %[[J]]]
//       CHECK:     pxa.reduce assign {{.*}}[%[[I]], %[[J]]]
func @mod(%arg0: memref<3x4xf16>) -> memref<3x4xf16> {
  %0 = alloc() : memref<3x4xf16>
  %1 = affine.parallel (%i, %j) = (0, 0) to (3, 4) reduce ("assign") -> (memref<3x4xf16>) {
    %2 = pxa.load %arg0[%i mod 3, %j mod 4] : memref<3x4xf16>
    %3 = pxa.reduce assign %2, %0[%i mod 3, %j mod 4] : memref<3x4xf16>
    affine.yield %3 : memref<3x4xf16>
  }
  return %1 : memref<3x4xf16>
}

// CHECK-LABEL: func @floordiv
//       CHECK:     pxa.load {{.*}}[0, 0]
//       CHECK:     pxa.reduce assign {{.*}}[0, 0]
func @floordiv(%arg0: memref<3x4xf16>) -> memref<3x4xf16> {
  %0 = alloc() : memref<3x4xf16>
  %1 = affine.parallel (%i, %j) = (0, 0) to (3, 4) reduce ("assign") -> (memref<3x4xf16>) {
    %2 = pxa.load %arg0[%i floordiv 3, %j floordiv 4] : memref<3x4xf16>
    %3 = pxa.reduce assign %2, %0[%i floordiv 3, %j floordiv 4] : memref<3x4xf16>
    affine.yield %3 : memref<3x4xf16>
  }
  return %1 : memref<3x4xf16>
}

// CHECK-LABEL: func @blocked_layout
//       CHECK:   affine.parallel (%[[I:[a-zA-Z0-9]*]], %[[J:[a-zA-Z0-9]*]])
//       CHECK:     affine.parallel (%[[K:[a-zA-Z0-9]*]], %[[L:[a-zA-Z0-9]*]])
//       CHECK:       pxa.load {{.*}}[%[[I]], %[[J]], %[[K]], %[[L]]]
func @blocked_layout(%arg0: memref<2x2x3x4xf16>) -> memref<6x8xf16> {
  %0 = alloc() : memref<6x8xf16>
  %1 = affine.parallel (%i, %j) = (0, 0) to (2, 2) reduce ("assign") -> (memref<6x8xf16>) {
    %2 = affine.parallel (%k, %l) = (0, 0) to (3, 4) reduce ("assign") -> (memref<6x8xf16>) {
      %3 = pxa.load %arg0[(%i * 3 + %k) floordiv 3,
                          (%j * 4 + %l) floordiv 4,
                          (%i * 3 + %k) mod 3,
                          (%j * 4 + %l) mod 4] : memref<2x2x3x4xf16>
      %4 = pxa.reduce assign %3, %0[%i * 3 + %k, %j * 4 + %l] : memref<6x8xf16>
      affine.yield %4 : memref<6x8xf16>
    }
    affine.yield %2 : memref<6x8xf16>
  }
  return %1 : memref<6x8xf16>
}
