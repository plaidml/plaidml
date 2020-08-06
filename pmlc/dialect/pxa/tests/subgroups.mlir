func @convolution(%arg0: memref<3x3x64x64xf32>, %arg1: memref<1x58x58x64xf32>) -> memref<1x56x56x64xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = alloc() : memref<1x56x56x64xf32>
  %1 = affine.parallel (%arg2, %arg3, %arg4) = (0, 0, 0) to (56, 56, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
    %2 = pxa.reduce assign %cst, %0[0, %arg2, %arg3, %arg4] : memref<1x56x56x64xf32>
    affine.yield %2 : memref<1x56x56x64xf32>
  }
  %2 = affine.parallel (%arg2, %arg3, %arg4, %arg5, %arg6, %arg7) = (0, 0, 0, 0, 0, 0) to (56, 56, 64, 3, 3, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
    %3 = affine.load %arg1[0, %arg2 + %arg5, %arg3 + %arg6, %arg7] : memref<1x58x58x64xf32>
    %4 = affine.load %arg0[%arg5, %arg6, %arg7, %arg4] : memref<3x3x64x64xf32>
    %5 = mulf %3, %4 : f32
    %6 = pxa.reduce addf %5, %1[0, %arg2, %arg3, %arg4] : memref<1x56x56x64xf32>
    affine.yield %6 : memref<1x56x56x64xf32>
  }
  return %2 : memref<1x56x56x64xf32>
}
