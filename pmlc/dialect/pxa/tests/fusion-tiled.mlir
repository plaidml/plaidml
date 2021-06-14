// RUN: pmlc-opt -pxa-fusion="tiled-fusion=true" -pxa-normalize -canonicalize %s | FileCheck %s

func @resnet_layer(%arg0: memref<1x56x56x64xf32>, %arg1: memref<1x1x64x64xf32>, %arg2: memref<64xf32>) -> memref<1x58x58x64xf32> {
  %cst = constant 0.0 : f32
  %0 = memref.alloc() : memref<1x56x56x64xf32>
  %1 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (56, 56, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
    %8 = pxa.reduce assign %cst, %0[0, %arg3, %arg4, %arg5] : memref<1x56x56x64xf32>
    affine.yield %8 : memref<1x56x56x64xf32>
  }
  %2 = affine.parallel (%arg3, %arg4) = (0, 0) to (56, 2) reduce ("assign") -> (memref<1x56x56x64xf32>) {
    %8 = affine.parallel (%i, %j, %k) = (0, 0, 0) to (56, 32, 64) reduce ("assign") -> memref<1x56x56x64xf32> {
      %a = affine.load %arg0[0, %i, %arg3, %k] : memref<1x56x56x64xf32>
      %b = affine.load %arg1[0, 0, %k, %j + %arg4 * 32] : memref<1x1x64x64xf32>
      %c = mulf %a, %b : f32
      %d = pxa.reduce addf %c, %1[0, %i, %arg3, %j + %arg4 * 32] : memref<1x56x56x64xf32>
      affine.yield %d : memref<1x56x56x64xf32>
    }
    affine.yield %8 : memref<1x56x56x64xf32>
  }
  %3 = memref.alloc() : memref<1x56x56x64xf32>
  %4 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (56, 56, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
    %8 = pxa.load %2[0, %arg3, %arg4, %arg5] : memref<1x56x56x64xf32>
    %9 = pxa.load %arg2[%arg5] : memref<64xf32>
    %10 = addf %8, %9 : f32
    %11 = pxa.reduce assign %10, %3[0, %arg3, %arg4, %arg5] : memref<1x56x56x64xf32>
    affine.yield %11 : memref<1x56x56x64xf32>
  }
  %5 = memref.alloc() : memref<1x58x58x64xf32>
  %6 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (58, 58, 64) reduce ("assign") -> (memref<1x58x58x64xf32>) {
    %8 = pxa.reduce assign %cst, %5[0, %arg3, %arg4, %arg5] : memref<1x58x58x64xf32>
    affine.yield %8 : memref<1x58x58x64xf32>
  }
  %7 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (56, 56, 64) reduce ("assign") -> (memref<1x58x58x64xf32>) {
    %8 = pxa.load %4[0, %arg3, %arg4, %arg5] : memref<1x56x56x64xf32>
    %9 = stdx.relu(%8) : (f32) -> f32
    %10 = pxa.reduce assign %9, %6[0, %arg3 + 1, %arg4 + 1, %arg5] : memref<1x58x58x64xf32>
    affine.yield %10 : memref<1x58x58x64xf32>
  }
  return %7 : memref<1x58x58x64xf32>
}
// CHECK-LABEL: func @resnet_layer
//       CHECK:   affine.parallel ({{.*}}, {{.*}}, {{.*}}) = (0, 0, 0) to (58, 58, 64)
//       CHECK:   affine.parallel ({{.*}}, {{.*}}) = (0, 0) to (56, 2)
//       CHECK:     affine.parallel ({{.*}}) = (0) to (56)
//       CHECK:       affine.parallel ({{.*}}) = (0) to (32)
//       CHECK:     affine.parallel ({{.*}}, {{.*}}, {{.*}}) = (0, 0, 0) to (56, 32, 64)
//       CHECK:     affine.parallel ({{.*}}) = (0) to (56)
//       CHECK:       affine.parallel ({{.*}}) = (0) to (32)
//       CHECK:     affine.parallel ({{.*}}) = (0) to (56)
//       CHECK:       affine.parallel ({{.*}}) = (0) to (32)
