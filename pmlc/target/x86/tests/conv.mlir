// PLAIDML_VERBOSE=5 bazel-bin/pmlc/opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true make-user-layouts-explicit=true" -canonicalize -x86-affine-stencil-xsmm pmlc/target/x86/tests/conv.mlir > output

func @cov() {
    %cst_0 = constant 0.000000e+00 : f32

    %31 = alloc() : memref<1x58x58x64xf32> // Input
    %arg88 = alloc():  memref<3x3x64x64xf32> // Filter
    %32 = alloc() : memref<1x56x56x64xf32> // output

    %arg3 = alloc() : memref<64xf32> // bias
    %arg12 = alloc() : memref<1x1x64x256xf32> // filter

    %33 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
      %637 = pxa.reduce assign %cst_0, %32[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      affine.yield %637 : memref<1x56x56x64xf32>
    }

    // CONV1
    %34 = affine.parallel (%arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117) = (0, 0, 0, 0, 0, 0, 0) to (1, 56, 56, 64, 3, 3, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
      %637 = pxa.load %31[%arg111, %arg112 + %arg115, %arg113 + %arg116, %arg117] : memref<1x58x58x64xf32>
      %638 = pxa.load %arg88[%arg115, %arg116, %arg117, %arg114] : memref<3x3x64x64xf32>
      %639 = mulf %637, %638 : f32
      %640 = pxa.reduce addf %639, %33[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      affine.yield %640 : memref<1x56x56x64xf32>
    }

    %35 = alloc() : memref<1x56x56x64xf32>
    %36 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
      %637 = pxa.load %34[0, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      %638 = pxa.load %arg3[%arg114] : memref<64xf32>
      %639 = addf %637, %638 : f32
      %640 = pxa.reduce assign %639, %35[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      affine.yield %640 : memref<1x56x56x64xf32>
    }
    %37 = alloc() : memref<1x56x56x64xi1>
    %38 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 64) reduce ("assign") -> (memref<1x56x56x64xi1>) {
      %637 = pxa.load %36[0, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      %638 = cmpf olt, %637, %cst_0 : f32
      %639 = pxa.reduce assign %638, %37[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x64xi1>
      affine.yield %639 : memref<1x56x56x64xi1>
    }
    %39 = alloc() : memref<1x56x56x64xf32>
    %40 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
      %637 = pxa.load %36[0, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      %638 = mulf %637, %cst_0 : f32
      %639 = pxa.reduce assign %638, %39[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      affine.yield %639 : memref<1x56x56x64xf32>
    }
    %41 = alloc() : memref<1x56x56x64xf32>
    %42 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
      %637 = pxa.load %38[0, %arg112, %arg113, %arg114] : memref<1x56x56x64xi1>
      %638 = pxa.load %40[0, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      %639 = pxa.load %36[0, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      %640 = select %637, %638, %639 : f32
      %641 = pxa.reduce assign %640, %41[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      affine.yield %641 : memref<1x56x56x64xf32>
    }
    %43 = alloc() : memref<1x56x56x256xf32>
    %44 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 256) reduce ("assign") -> (memref<1x56x56x256xf32>) {
      %637 = pxa.reduce assign %cst_0, %43[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x256xf32>
      affine.yield %637 : memref<1x56x56x256xf32>
    }

   // CONV2
    %45 = affine.parallel (%arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117) = (0, 0, 0, 0, 0, 0, 0) to (1, 56, 56, 256, 1, 1, 64) reduce ("assign") -> (memref<1x56x56x256xf32>) {
      %637 = pxa.load %42[%arg111, %arg112 + %arg115, %arg113 + %arg116, %arg117] : memref<1x56x56x64xf32>
      %638 = pxa.load %arg12[%arg115, %arg116, %arg117, %arg114] : memref<1x1x64x256xf32>
      %639 = mulf %637, %638 : f32
      %640 = pxa.reduce addf %639, %44[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x256xf32>
      affine.yield %640 : memref<1x56x56x256xf32>
    }


   return
}
