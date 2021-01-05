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
  func @matrixPower(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>) -> memref<16x16xf32> {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c4 = constant 4 : index
    %0 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %arg0) -> (memref<16x16xf32>) {
      %1 = alloc() : memref<16x16xf32>
      %2 = affine.parallel (%arg4, %arg5) = (0, 0) to (16, 16) reduce ("assign") -> (memref<16x16xf32>) {
        %7 = pxa.reduce assign %cst, %1[%arg4, %arg5] : memref<16x16xf32>
        affine.yield %7 : memref<16x16xf32>
      }
      %3 = affine.parallel (%arg4, %arg5, %arg6) = (0, 0, 0) to (16, 16, 16) reduce ("assign") -> (memref<16x16xf32>) {
        %7 = pxa.load %arg3[%arg4, %arg6] : memref<16x16xf32>
        %8 = pxa.load %arg0[%arg6, %arg5] : memref<16x16xf32>
        %9 = mulf %7, %8 : f32
        %10 = pxa.reduce addf %9, %2[%arg4, %arg5] : memref<16x16xf32>
        affine.yield %10 : memref<16x16xf32>
      }
      %4 = alloc() : memref<16x16xf32>
      %5 = affine.parallel (%arg4, %arg5) = (0, 0) to (16, 16) reduce ("assign") -> (memref<16x16xf32>) {
        %7 = pxa.reduce assign %cst, %4[%arg4, %arg5] : memref<16x16xf32>
        affine.yield %7 : memref<16x16xf32>
      }
      %6 = affine.parallel (%arg4, %arg5, %arg6) = (0, 0, 0) to (16, 16, 16) reduce ("assign") -> (memref<16x16xf32>) {
        %7 = pxa.load %3[%arg4, %arg6] : memref<16x16xf32>
        %8 = pxa.load %arg0[%arg6, %arg5] : memref<16x16xf32>
        %9 = mulf %7, %8 : f32
        %10 = pxa.reduce addf %9, %5[%arg4, %arg5] : memref<16x16xf32>
        affine.yield %10 : memref<16x16xf32>
      }
      scf.yield %6 : memref<16x16xf32>
    }
    return %0 : memref<16x16xf32>
  }
}

// CHECK-LABEL: @matrixPowers
