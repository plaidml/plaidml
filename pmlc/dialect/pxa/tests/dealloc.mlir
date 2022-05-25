// RUN: pmlc-opt -split-input-file -pxa-dealloc-placement %s | FileCheck %s

func.func @double_for(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>) -> memref<16x16xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %arg0) -> (memref<16x16xf32>) {
    %1 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg5 = %arg3) -> (memref<16x16xf32>) {
      %2 = memref.alloc() : memref<16x16xf32>
      %3 = affine.parallel (%arg6, %arg7) = (0, 0) to (16, 16) reduce ("assign") -> (memref<16x16xf32>) {
        %5 = pxa.reduce assign %cst, %2[%arg6, %arg7] : memref<16x16xf32>
        affine.yield %5 : memref<16x16xf32>
      }
      %4 = affine.parallel (%arg6, %arg7, %arg8) = (0, 0, 0) to (16, 16, 16) reduce ("assign") -> (memref<16x16xf32>) {
        %5 = pxa.load %arg5[%arg6, %arg8] : memref<16x16xf32>
        %6 = pxa.load %arg0[%arg8, %arg7] : memref<16x16xf32>
        %7 = arith.mulf %5, %6 : f32
        %8 = pxa.reduce addf %7, %3[%arg6, %arg7] : memref<16x16xf32>
        affine.yield %8 : memref<16x16xf32>
      }
      scf.yield %4 : memref<16x16xf32>
    }
    scf.yield %1 : memref<16x16xf32>
  }
  return %0 : memref<16x16xf32>
}
// CHECK-LABEL: func.func @double_for
//  CHECK-SAME: (%[[arg0:.*]]: memref<16x16xf32>, %[[arg1:.*]]: memref<16x16xf32>)
//       CHECK:   %[[t0:.*]] = memref.alloc() : memref<16x16xf32>
//       CHECK:   %[[t1:.*]] = affine.parallel (%[[arg2:.*]], %[[arg3:.*]]) = (0, 0) to (16, 16)
//       CHECK:     %[[t2:.*]] = pxa.load %[[arg0]][%[[arg2]], %[[arg3]]]
//       CHECK:     %[[t3:.*]] = pxa.reduce assign %[[t2]], %[[t0]][%[[arg2]], %[[arg3]]]
//       CHECK:     affine.yield %[[t3]]
//       CHECK:   scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args(%[[arg3:.*]] = %[[t1]])
//       CHECK:     %[[t4:.*]] = memref.alloc() : memref<16x16xf32>
//       CHECK:     %[[t5:.*]] = affine.parallel (%[[arg4:.*]], %[[arg5:.*]]) = (0, 0) to (16, 16)
//       CHECK:       %[[t6:.*]] = pxa.load %[[arg3]][%[[arg4]], %[[arg5]]]
//       CHECK:       %[[t7:.*]] = pxa.reduce assign %[[t6]], %[[t4]][%[[arg4]], %[[arg5]]]
//       CHECK:       affine.yield %[[t7]]
//       CHECK:     memref.dealloc %[[arg3]]
//       CHECK:     scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args(%[[arg6:.*]] = %[[t5]])
//       CHECK:       memref.alloc()
//       CHECK:       affine.parallel
//       CHECK:       affine.parallel
//       CHECK:       memref.dealloc %[[arg6]]

// -----

func.func @matrix_power(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>) -> memref<16x16xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %arg0) -> (memref<16x16xf32>) {
    %1 = memref.alloc() : memref<16x16xf32>
    %2 = affine.parallel (%arg4, %arg5) = (0, 0) to (16, 16) reduce ("assign") -> (memref<16x16xf32>) {
      %7 = pxa.reduce assign %cst, %1[%arg4, %arg5] : memref<16x16xf32>
      affine.yield %7 : memref<16x16xf32>
    }
    %3 = affine.parallel (%arg4, %arg5, %arg6) = (0, 0, 0) to (16, 16, 16) reduce ("assign") -> (memref<16x16xf32>) {
      %7 = pxa.load %arg3[%arg4, %arg6] : memref<16x16xf32>
      %8 = pxa.load %arg0[%arg6, %arg5] : memref<16x16xf32>
      %9 = arith.mulf %7, %8 : f32
      %10 = pxa.reduce addf %9, %2[%arg4, %arg5] : memref<16x16xf32>
      affine.yield %10 : memref<16x16xf32>
    }
    %4 = memref.alloc() : memref<16x16xf32>
    %5 = affine.parallel (%arg4, %arg5) = (0, 0) to (16, 16) reduce ("assign") -> (memref<16x16xf32>) {
      %7 = pxa.reduce assign %cst, %4[%arg4, %arg5] : memref<16x16xf32>
      affine.yield %7 : memref<16x16xf32>
    }
    %6 = affine.parallel (%arg4, %arg5, %arg6) = (0, 0, 0) to (16, 16, 16) reduce ("assign") -> (memref<16x16xf32>) {
      %7 = pxa.load %3[%arg4, %arg6] : memref<16x16xf32>
      %8 = pxa.load %arg0[%arg6, %arg5] : memref<16x16xf32>
      %9 = arith.mulf %7, %8 : f32
      %10 = pxa.reduce addf %9, %5[%arg4, %arg5] : memref<16x16xf32>
      affine.yield %10 : memref<16x16xf32>
    }
    scf.yield %6 : memref<16x16xf32>
  }
  return %0 : memref<16x16xf32>
}

// CHECK-LABEL: func.func @matrix_power
// CHECK-SAME: (%[[arg0:.*]]: memref<16x16xf32>, %[[arg1:.*]]: memref<16x16xf32>)
//      CHECK:   %[[t0:.*]] = memref.alloc() : memref<16x16xf32>
//      CHECK:   %[[t1:.*]] = affine.parallel (%[[arg2:.*]], %[[arg3:.*]]) = (0, 0) to (16, 16)
//      CHECK:     %[[t2:.*]] = pxa.load %[[arg0]][%[[arg2]], %[[arg3]]]
//      CHECK:     %[[t3:.*]] = pxa.reduce assign %[[t2]], %[[t0]][%[[arg2]], %[[arg3]]]
//      CHECK:     affine.yield %[[t3]]
//      CHECK:   scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args(%[[arg3:.*]] = %[[t1]])
//      CHECK:     %[[a0:.*]] = memref.alloc
//      CHECK:     affine.parallel
//      CHECK:     %[[t4:.*]] = affine.parallel
//      CHECK:     memref.dealloc %[[arg3]]
//      CHECK:     memref.alloc
//      CHECK:     affine.parallel
//      CHECK:     affine.parallel
//      CHECK:     memref.dealloc %[[a0]]

// -----

#map0 = affine_map<(d0, d1, d2) -> (0, d0, 0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (0, d0, 0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (0, 0, d2, d1)>

func.func @post_fusion(%arg0: memref<16xf32>, %arg1: memref<1x1x32x16xf32>, %arg2: memref<1x1x16x96xf32>, %arg3: memref<96xf32>, %arg4: memref<1x112x112x32xf32>, %arg5: memref<1x112x112x96xf32>) -> memref<1x112x112x96xf32> {
  %8 = memref.alloc() : memref<1x112x112x16xf32>
  %9 = affine.parallel (%arg109) = (0) to (8) reduce ("assign") -> (memref<1x112x112x16xf32>) {
    %148 = affine.parallel (%arg110, %arg111, %arg112) = (0, 0, 0) to (112, 112, 2) reduce ("assign") -> (memref<1x112x112x16xf32>) {
      %149 = pxa.load %arg0[%arg112 + %arg109 * 2] : memref<16xf32>
      %150 = pxa.reduce assign %149, %8[0, %arg110, %arg111, %arg112 + %arg109 * 2] : memref<1x112x112x16xf32>
      affine.yield %150 : memref<1x112x112x16xf32>
    }
    affine.yield %148 : memref<1x112x112x16xf32>
  }
  %11 = affine.parallel (%arg109) = (0) to (8) reduce ("assign") -> (memref<1x112x112x96xf32>) {
    %148 = affine.parallel (%arg110, %arg111, %arg112) = (0, 0, 0) to (112, 112, 12) reduce ("assign") -> (memref<1x112x112x96xf32>) {
      %149 = pxa.load %arg3[%arg112 + %arg109 * 12] : memref<96xf32>
      %150 = pxa.reduce assign %149, %arg5[0, %arg110, %arg111, %arg112 + %arg109 * 12] : memref<1x112x112x96xf32>
      affine.yield %150 : memref<1x112x112x96xf32>
    }
    affine.yield %148 : memref<1x112x112x96xf32>
  }
  %12 = affine.parallel (%arg109) = (0) to (8) reduce ("assign") -> (memref<1x112x112x96xf32>) {
    %148 = affine.parallel (%arg110, %arg111) = (0, 0) to (2, 14) reduce ("assign") -> (memref<1x112x112x16xf32>) {
      %150 = pxa.generic (%9[0, %arg110 * 56, %arg111 + %arg109 * 14, 0]: #map0) <addf> @tpp_gemm(%arg4[0, %arg110 * 56, %arg111 + %arg109 * 14, 0]: #map1, %arg1[0, 0, 0, 0]: #map2) tile: [56, 16, 32] : (memref<1x112x112x32xf32>, memref<1x1x32x16xf32>) -> memref<1x112x112x16xf32>
      affine.yield %150 : memref<1x112x112x16xf32>
    }
    %149 = affine.parallel (%arg110, %arg111) = (0, 0) to (7, 14) reduce ("assign") -> (memref<1x112x112x96xf32>) {
      %150 = pxa.generic (%11[0, %arg110 * 16, %arg111 + %arg109 * 14, 0]: #map0) <addf> @tpp_gemm(%148[0, %arg110 * 16, %arg111 + %arg109 * 14, 0]: #map1, %arg2[0, 0, 0, 0]: #map2) tile: [16, 96, 16] : (memref<1x112x112x16xf32>, memref<1x1x16x96xf32>) -> memref<1x112x112x96xf32>
      affine.yield %150 : memref<1x112x112x96xf32>
    }
    affine.yield %149 : memref<1x112x112x96xf32>
  }
  return %12 : memref<1x112x112x96xf32>
}

// CHECK-LABEL: func.func @post_fusion
//       CHECK:   %[[alloc:.*]] = memref.alloc
//       CHECK:   affine.parallel
//       CHECK:     affine.parallel
//       CHECK:   affine.parallel
//       CHECK:     affine.parallel
//       CHECK:   affine.parallel
//       CHECK:     affine.parallel
//       CHECK:     affine.parallel
//       CHECK:   memref.dealloc %[[alloc]]
