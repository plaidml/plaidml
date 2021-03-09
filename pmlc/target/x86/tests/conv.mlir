
func @cov() {
    %cst_0 = constant 0.000000e+00 : f32

    %31 = alloc() : memref<1x58x58x64xf32> // Input
    %arg88 = alloc():  memref<3x3x64x64xf32> // Filter

    %32 = alloc() : memref<1x56x56x64xf32>
    %33 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
      %637 = pxa.reduce assign %cst_0, %32[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      affine.yield %637 : memref<1x56x56x64xf32>
    }
    %34 = affine.parallel (%arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117) = (0, 0, 0, 0, 0, 0, 0) to (1, 56, 56, 64, 3, 3, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
      %637 = pxa.load %31[%arg111, %arg112 + %arg115, %arg113 + %arg116, %arg117] : memref<1x58x58x64xf32>
      %638 = pxa.load %arg88[%arg115, %arg116, %arg117, %arg114] : memref<3x3x64x64xf32>
      %639 = mulf %637, %638 : f32
      %640 = pxa.reduce addf %639, %33[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      affine.yield %640 : memref<1x56x56x64xf32>
    }

   return
}
