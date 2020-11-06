// RUN: pmlc-opt -pxa-reorder-layouts -pxa-simplify-with-constraints %s | FileCheck %s -check-prefix=CHECK -check-prefix="NORDR"
// RUN: pmlc-opt -pxa-reorder-layouts="allow-reorder=true" -pxa-simplify-with-constraints %s | FileCheck %s -check-prefix=CHECK -check-prefix="RDR"

// CHECK-LABEL: func @permute_layout
//       CHECK:   affine.parallel (%[[I:[a-zA-Z0-9]*]], %[[J:[a-zA-Z0-9]*]])
//       CHECK:     pxa.reduce assign {{.*}}[%[[J]], %[[I]]]
//       CHECK:   affine.parallel (%[[I:[a-zA-Z0-9]*]], %[[J:[a-zA-Z0-9]*]])
//       CHECK:     pxa.load {{.*}}[%[[I]], %[[J]]]
func @permute_layout() -> memref<1xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = alloc() : memref<2x3xf32>
  %1 = affine.parallel (%i, %j) = (0, 0) to (2, 3) reduce("assign") -> memref<2x3xf32> {
    %2 = pxa.reduce assign %cst, %0[%i, %j] : memref<2x3xf32>
    affine.yield %2 : memref<2x3xf32>
  }
  %2 = alloc() : memref<1xf32>
  %3 = affine.parallel (%i, %j) = (0, 0) to (3, 2) reduce("addf") -> memref<1xf32> {
    %4 = pxa.load %1[%j, %i] : memref<2x3xf32>
    %5 = pxa.reduce addf %4, %2[0] : memref<1xf32>
    affine.yield %5 : memref<1xf32>
  }
  return %3 : memref<1xf32>
}

// CHECK-LABEL: func @block_layout
//       CHECK:   affine.parallel (%[[I:[a-zA-Z0-9]*]], %[[J:[a-zA-Z0-9]*]])
//       CHECK:     pxa.reduce assign %{{.*}}[%[[I]] floordiv 4, %[[J]] floordiv 4, %[[I]] mod 4, %[[J]] mod 4]
//       CHECK:   affine.parallel (%[[I:[a-zA-Z0-9]*]], %[[J:[a-zA-Z0-9]*]])
//       CHECK:     affine.parallel (%[[K:[a-zA-Z0-9]*]], %[[L:[a-zA-Z0-9]*]])
//       CHECK:       pxa.load {{.*}}[%[[I]], %[[J]], %[[K]], %[[L]]]
func @block_layout() -> memref<1xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = alloc() : memref<16x16xf32>
  %1 = affine.parallel (%i, %j) = (0, 0) to (16, 16) reduce("assign") -> memref<16x16xf32> {
    %2 = pxa.reduce assign %cst, %0[%i, %j] : memref<16x16xf32>
    affine.yield %2 : memref<16x16xf32>
  }
  %2 = alloc() : memref<1xf32>
  %3 = affine.parallel (%i, %j) = (0, 0) to (4, 4) reduce("assign") -> memref<1xf32> {
    %4 = affine.parallel (%k, %l) = (0, 0) to (4, 4) reduce("addf") -> memref<1xf32> {
      %5 = pxa.load %1[%i * 4 + %k, %j * 4 + %l] : memref<16x16xf32>
      %6 = pxa.reduce addf %5, %2[0] : memref<1xf32>
      affine.yield %6 : memref<1xf32>
    }
    affine.yield %4 : memref<1xf32>
  }
  return %3 : memref<1xf32>
}

// CHECK-LABEL: func @separate_reorder
//  CHECK-SAME:     %[[ARG:[a-zA-Z0-9]*]]:
//         RDR:   %[[ALLOC:.*]] = alloc() : memref<4x4x4x4xf32>
//         RDR:   %[[NEW:.*]] = affine.parallel (%[[I:[a-zA-Z0-9]*]], %[[J:[a-zA-Z0-9]*]])
//         RDR:     pxa.load %[[ARG]][%[[I]], %[[J]]]
//         RDR:     pxa.reduce assign {{.*}}, %[[ALLOC]][%[[I]] floordiv 4, %[[J]] floordiv 4, %[[I]] mod 4, %[[J]] mod 4]
//       CHECK:   affine.parallel (%[[I:[a-zA-Z0-9]*]], %[[J:[a-zA-Z0-9]*]])
//       CHECK:     affine.parallel (%[[K:[a-zA-Z0-9]*]], %[[L:[a-zA-Z0-9]*]])
//       NORDR:       pxa.load %[[ARG]][%[[I]] * 4 + %[[K]], %[[J]] * 4 + %[[L]]]
//         RDR:       pxa.load %[[NEW]][%[[I]], %[[J]], %[[K]], %[[L]]]
func @separate_reorder(%arg0: memref<16x16xf32>) -> memref<1xf32> {
  %0 = alloc() : memref<1xf32>
  %1 = affine.parallel (%i, %j) = (0, 0) to (4, 4) reduce("assign") -> memref<1xf32> {
    %2 = affine.parallel (%k, %l) = (0, 0) to (4, 4) reduce("addf") -> memref<1xf32> {
      %3 = pxa.load %arg0[%i * 4 + %k, %j * 4 + %l] : memref<16x16xf32>
      %4 = pxa.reduce addf %3, %0[0] : memref<1xf32>
      affine.yield %4 : memref<1xf32>
    }
    affine.yield %2 : memref<1xf32>
  }
  return %1 : memref<1xf32>
}

// CHECK-LABEL: func @vector_load
//       CHECK:   affine.parallel (%[[I:[a-zA-Z0-9]*]], %[[J:[a-zA-Z0-9]*]])
//       CHECK:     pxa.reduce assign %{{.*}}[%[[J]] floordiv 4, %[[I]], %[[J]] mod 4]
//       CHECK:   affine.parallel (%[[I:[a-zA-Z0-9]*]], %[[J:[a-zA-Z0-9]*]])
//       CHECK:     pxa.vector_load {{.*}}[%[[I]], %[[J]], 0]
func @vector_load() -> memref<4xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = alloc() : memref<2x16xf32>
  %1 = affine.parallel (%i, %j) = (0, 0) to (2, 16) reduce("assign") -> memref<2x16xf32> {
    %2 = pxa.reduce assign %cst, %0[%i, %j] : memref<2x16xf32>
    affine.yield %2 : memref<2x16xf32>
  }
  %2 = alloc() : memref<4xf32>
  %3 = affine.parallel (%i, %j) = (0, 0) to (4, 2) reduce("assign") -> memref<4xf32> {
    %4 = pxa.vector_load %1[%j, %i * 4] : memref<2x16xf32>, vector<4xf32>
    %5 = pxa.vector_reduce addf %4, %2[0] : memref<4xf32>, vector<4xf32>
    affine.yield %5 : memref<4xf32>
  }
  return %3 : memref<4xf32>
}

// CHECK-LABEL: func @vector_store
//       CHECK:   affine.parallel (%[[I:[a-zA-Z0-9]*]], %[[J:[a-zA-Z0-9]*]])
//       CHECK:     pxa.vector_reduce assign %{{.*}}[%[[J]], %[[I]], 0]
//       CHECK:   affine.parallel (%[[I:[a-zA-Z0-9]*]], %[[J:[a-zA-Z0-9]*]])
//       CHECK:     pxa.load {{.*}}[%[[I]] floordiv 4, %[[J]], %[[I]] mod 4]
func @vector_store() -> memref<1xf32> {
  %cst = constant 0.000000e+00 : f32
  %cst_vec = vector.broadcast %cst : f32 to vector<4xf32>
  %0 = alloc() : memref<2x16xf32>
  %1 = affine.parallel (%i, %j) = (0, 0) to (2, 4) reduce("assign") -> memref<2x16xf32> {
    %2 = pxa.vector_reduce assign %cst_vec, %0[%i, %j * 4] : memref<2x16xf32>, vector<4xf32>
    affine.yield %2 : memref<2x16xf32>
  }
  %2 = alloc() : memref<1xf32>
  %3 = affine.parallel (%i, %j) = (0, 0) to (16, 2) reduce("assign") -> memref<1xf32> {
    %4 = pxa.load %1[%j, %i] : memref<2x16xf32>
    %5 = pxa.reduce addf %4, %2[0] : memref<1xf32>
    affine.yield %5 : memref<1xf32>
  }
  return %3 : memref<1xf32>
}

// CHECK-LABEL: func @different_vector
//       CHECK:   affine.parallel (%[[I:[a-zA-Z0-9]*]])
//       CHECK:     pxa.vector_reduce assign %{{.*}}[0, %[[I]]]
//       CHECK:   affine.parallel (%[[I:[a-zA-Z0-9]*]], %[[J:[a-zA-Z0-9]*]])
//       CHECK:     pxa.vector_load {{.*}}[%[[J]], %[[I]] * 4]
func @different_vector() -> memref<4xf32> {
  %cst = constant 0.000000e+00 : f32
  %cst_vec = vector.broadcast %cst : f32 to vector<2x1xf32>
  %0 = alloc() : memref<2x16xf32>
  %1 = affine.parallel (%i) = (0) to (16) reduce("assign") -> memref<2x16xf32> {
    %2 = pxa.vector_reduce assign %cst_vec, %0[0, %i] : memref<2x16xf32>, vector<2x1xf32>
    affine.yield %2 : memref<2x16xf32>
  }
  %2 = alloc() : memref<4xf32>
  %3 = affine.parallel (%i, %j) = (0, 0) to (4, 2) reduce("assign") -> memref<4xf32> {
    %4 = pxa.vector_load %1[%j, %i * 4] : memref<2x16xf32>, vector<4xf32>
    %5 = pxa.vector_reduce addf %4, %2[0] : memref<4xf32>, vector<4xf32>
    affine.yield %5 : memref<4xf32>
  }
  return %3 : memref<4xf32>
}
