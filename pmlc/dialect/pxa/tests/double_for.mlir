// RUN: pmlc-opt -pxa-dealloc-placement %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0, 0)>
#map2 = affine_map<() -> (16, 16)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<() -> (0, 0, 0)>
#map7 = affine_map<() -> (16, 16, 16)>

module {
// CHECK-LABEL: @doubleFor
  func @doubleFor(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>) -> memref<16x16xf32> {
// CHECK-SAME: (%[[arg0:.*]]: memref<16x16xf32>, %[[arg1:.*]]: memref<16x16xf32>)
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c4 = constant 4 : index
// CHECK:   %[[t0:.*]] = alloc() : memref<16x16xf32>
// CHECK:   %[[t1:.*]] = affine.parallel (%[[arg2:.*]], %[[arg3:.*]]) = (0, 0) to (16, 16)
// CHECK:     %[[t2:.*]] = pxa.load %[[arg0]][%[[arg2]], %[[arg3]]]
// CHECK:     %[[t3:.*]] = pxa.reduce assign %[[t2]], %[[t0]][%[[arg2]], %[[arg3]]]
// CHECK:     affine.yield %[[t3]]
    %0 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %arg0) -> (memref<16x16xf32>) {
// CHECK:   scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args(%[[arg3:.*]] = %[[t1]])
// CHECK:     %[[t4:.*]] = alloc() : memref<16x16xf32>
// CHECK:     %[[t5:.*]] = affine.parallel (%[[arg4:.*]], %[[arg5:.*]]) = (0, 0) to (16, 16)
// CHECK:       %[[t6:.*]] = pxa.load %[[arg3]][%[[arg4]], %[[arg5]]]
// CHECK:       %[[t7:.*]] = pxa.reduce assign %[[t6]], %[[t4]][%[[arg4]], %[[arg5]]]
// CHECK:       affine.yield %[[t7]]
      %1 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg5 = %arg3) -> (memref<16x16xf32>) {
// CHECK:     dealloc %[[arg3]]
// CHECK:     scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args(%[[arg6:.*]] = %[[t5]])
        %2 = alloc() : memref<16x16xf32>
// CHECK:       alloc
        %3 = affine.parallel (%arg6, %arg7) = (0, 0) to (16, 16) reduce ("assign") -> (memref<16x16xf32>) {
// CHECK:       affine.parallel
          %5 = pxa.reduce assign %cst, %2[%arg6, %arg7] : memref<16x16xf32>
          affine.yield %5 : memref<16x16xf32>
        }
        %4 = affine.parallel (%arg6, %arg7, %arg8) = (0, 0, 0) to (16, 16, 16) reduce ("assign") -> (memref<16x16xf32>) {
// CHECK:       affine.parallel
          %5 = pxa.load %arg5[%arg6, %arg8] : memref<16x16xf32>
          %6 = pxa.load %arg0[%arg8, %arg7] : memref<16x16xf32>
          %7 = mulf %5, %6 : f32
          %8 = pxa.reduce addf %7, %3[%arg6, %arg7] : memref<16x16xf32>
          affine.yield %8 : memref<16x16xf32>
        }
// CHECK:       dealloc %[[arg6]]
        scf.yield %4 : memref<16x16xf32>
      }
      scf.yield %1 : memref<16x16xf32>
    }
    return %0 : memref<16x16xf32>
  }
}
