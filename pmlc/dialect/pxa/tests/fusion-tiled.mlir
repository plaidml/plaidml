// RUN: pmlc-opt -pxa-fusion="tiled-fusion=true" -pxa-fusion="loop-depth=3" -pxa-normalize -canonicalize %s | FileCheck %s

func @fusion_tiled() -> memref<1x56x56x256xi1> {
  %cst = constant 0.000000e+00 : f32
  %0 = alloc() : memref<1x56x56x256xf32>
  %1 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (56, 56, 256) reduce ("assign") -> (memref<1x56x56x256xf32>) {
    %6 = pxa.reduce assign %cst, %0[0, %arg3, %arg4, %arg5] : memref<1x56x56x256xf32>
    affine.yield %6 : memref<1x56x56x256xf32>
  } {tags = {subgroupSize = 1 : i64}}
  %2 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (56, 7, 32) reduce ("assign") -> (memref<1x56x56x256xf32>) {
    %6 = alloc() : memref<1x1x8x1xvector<8xf32>>
    %7 = affine.parallel (%arg6) = (0) to (8) reduce ("assign") -> (memref<1x1x8x1xvector<8xf32>>) {
      %10 = vector.broadcast %cst : f32 to vector<8xf32>
      %11 = pxa.reduce assign %10, %6[0, 0, %arg6, 0] : memref<1x1x8x1xvector<8xf32>>
      affine.yield %11 : memref<1x1x8x1xvector<8xf32>>
    }
    %8 = affine.parallel (%arg6) = (0) to (8) reduce ("assign") -> (memref<1x56x56x256xf32>) {
      %9 = pxa.load %7[0, 0, %arg6, 0] : memref<1x1x8x1xvector<8xf32>>
      %10 = pxa.vector_reduce addf %9, %1[0, %arg3, %arg4 * 8 + %arg6, %arg5 * 8] : memref<1x56x56x256xf32>, vector<8xf32>
      affine.yield %10 : memref<1x56x56x256xf32>
    }
    affine.yield %8 : memref<1x56x56x256xf32>
  } {tags = {subgroupSize = 8 : i64}}
  %3 = alloc() : memref<1x56x56x256xi1>
  %4 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (56, 56, 256) reduce ("assign") -> (memref<1x56x56x256xi1>) {
    %6 = pxa.load %2[0, %arg3, %arg4, %arg5] : memref<1x56x56x256xf32>
    %7 = cmpf "olt", %6, %cst : f32
    %8 = pxa.reduce assign %7, %3[0, %arg3, %arg4, %arg5] : memref<1x56x56x256xi1>
    affine.yield %8 : memref<1x56x56x256xi1>
  } {tags = {subgroupSize = 1 : i64}}
  return %4 : memref<1x56x56x256xi1>
}

// CHECK-LABEL: func @fusion_tiled
// CHECK:       affine.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (0, 0, 0) to (56, 7, 32)
// CHECK-NOT:   affine.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (0, 0, 0) to (56, 56, 256)
